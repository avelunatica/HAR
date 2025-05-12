
from threading import Thread
import time
import cv2
import numpy as np
import ffmpeg
from loguru import logger
from core.inference_model import InferenceModel

BUFFER_SIZE = 5
FRAME_RATE = 20

class StreamActionRecognizer:
    def __init__(self, config_path, checkpoint_path, label_path, device, camera_url, validator=None, resize_width=600):
        self.model = InferenceModel(config_path, checkpoint_path, device, label_path)
        self.camera_url = camera_url
        self.validator = validator
        self.resize_width = resize_width
        self.running = True
        self.width = None
        self.height = None
        self.frame_queue = []
        self.display_queue = []

    def detect_resolution(self):
        logger.info("🔍 Detectando resolución do stream...")
        try:
            probe = ffmpeg.probe(
                self.camera_url,
                v='error',
                select_streams='v:0',
                show_entries='stream=width,height',
                rtsp_transport='tcp'
            )
            stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if not stream:
                raise ValueError("Non se atopou ningún stream de vídeo válido.")
            self.width = int(stream['width'])
            self.height = int(stream['height'])
            logger.success(f"🎯 Resolución detectada con ffprobe: {self.width}x{self.height}")
        except Exception as e:
            logger.warning(f"⚠️ ffprobe fallou: {e}")
            logger.info("🔄 Tentando con OpenCV...")
            cap = cv2.VideoCapture(self.camera_url)
            if not cap.isOpened():
                raise RuntimeError("❌ Non se pode abrir o stream con OpenCV.")
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            logger.success(f"✅ Resolución detectada con OpenCV: {self.width}x{self.height}")

    def capture_frames(self):
        self.detect_resolution()

        input_kwargs = {}
        if self.camera_url.startswith("rtsp://"):
            input_kwargs = {'rtsp_transport': 'tcp', 'f': 'rtsp'}
        elif self.camera_url.startswith("http://"):
            input_kwargs = {'f': 'mjpeg'}

        while self.running:
            logger.info("🎥 Iniciando proceso de captura con ffmpeg-python...")
            try:
                process = (
                    ffmpeg
                    .input(self.camera_url, **input_kwargs)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )

                while self.running:
                    in_bytes = process.stdout.read(self.width * self.height * 3)

                    if not in_bytes:
                        logger.warning("⚠️ No se recibieron datos del stream.")
                        break

                    frame = np.frombuffer(in_bytes, np.uint8)
                    if frame.size != self.width * self.height * 3:
                        logger.warning(f"⚠️ Tamaño de frame incorrecto: {frame.size}, esperado: {self.width * self.height * 3}.")
                        continue

                    frame = frame.reshape((self.height, self.width, 3))

                    if len(self.frame_queue) < BUFFER_SIZE:
                        self.frame_queue.append(frame)

                    if self.display_queue:
                        self.display_queue.pop(0)
                    self.display_queue.append(frame)

                    time.sleep(1 / FRAME_RATE)

            except Exception as e:
                logger.error(f"❌ Error en captura: {e}")
                break  # Exit the loop if an error occurs

            finally:
                if process:
                    process.stdout.close()
                    stderr = process.stderr.read()
                    if stderr:
                        logger.error(f"🐞 ffmpeg stderr:\n{stderr.decode(errors='ignore')}")
                    process.wait()

            logger.info("🔁 Reintentando conexión en 3 segundos...")
            time.sleep(3)

    def run_inference(self):
        while self.running:
            if len(self.frame_queue) >= BUFFER_SIZE:
                frames = list(self.frame_queue)
                self.frame_queue.clear()

                results = self.model.predict(frames)
                resultados_agrupados = {}

                for label, score in results:
                    if label != "---":
                        resultados_agrupados[label] = resultados_agrupados.get(label, 0) + score

                resultados_ordenados = sorted(resultados_agrupados.items(), key=lambda x: x[1], reverse=True)

                if resultados_ordenados:
                    logger.info("🔍 Accións detectadas:")
                    for label, score in resultados_ordenados:
                        logger.info(f"   ➜ {label}: {score * 100:.2f}%")

                    if self.validator:
                        validated = self.validator.update(resultados_ordenados)
                        if validated:
                            logger.success(f"📡 Acción validada e enviada: {validated}")
                else:
                    logger.warning("⚠️ Ningunha acción recoñecida neste bloque de frames.")

            else:
                time.sleep(0.05)


    def show_video(self):
        cv2.namedWindow("Stream Video", cv2.WINDOW_NORMAL)
        while self.running:
            if self.display_queue:
                frame = self.display_queue[0]
                h, w = frame.shape[:2]
                new_height = int((self.resize_width / w) * h)
                resized = cv2.resize(frame, (self.resize_width, new_height))
                frame_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                cv2.imshow("Stream Video", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("👋 Pechando ventá de vídeo.")
                self.running = False
                break

        cv2.destroyAllWindows()

    def start(self):
        logger.info("🚀 Lanzando StreamActionRecognizer...")

        capture_thread = Thread(target=self.capture_frames, daemon=True)
        inference_thread = Thread(target=self.run_inference, daemon=True)

        capture_thread.start()
        inference_thread.start()
        self.show_video()

        capture_thread.join()
        inference_thread.join()
        logger.info("🛑 Cliente detido completamente.")
