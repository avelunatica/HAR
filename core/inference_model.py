import torch
import numpy as np
from loguru import logger
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmaction.apis import init_recognizer
from mmaction.utils import get_str_type

EXCLUDED_STEPS = [
    "OpenCVInit", "OpenCVDecode", "DecordInit", "DecordDecode",
    "PyAVInit", "PyAVDecode", "RawFrameDecode"
]

class InferenceModel:
    def __init__(self, config_path, checkpoint_path, device, label_path):
        self.device = torch.device(device)
        self.cfg = Config.fromfile(config_path)
        self.model = init_recognizer(self.cfg, checkpoint_path, device=device)

        with open(label_path, "r") as f:
            self.labels = [line.strip() for line in f]

        self.test_pipeline = Compose(
            [s for s in self.cfg.test_pipeline if get_str_type(s["type"]) not in EXCLUDED_STEPS]
        )

        logger.success("🧠 Modelo de inferencia inicializado correctamente.")

    def predict(self, frames):
        data = dict(
            img_shape=frames[0].shape[:2],
            modality="RGB",
            label=-1,
            total_frames=1,
            start_index=0,
            imgs=frames
        )
        data = self.test_pipeline(data)
        data = pseudo_collate([data])

        with torch.no_grad():
            result = self.model.test_step(data)[0]

        scores = np.array(result.pred_score.tolist())
        logger.info("🔮 Predicción realizada sobre frames.")
        return sorted(zip(self.labels, scores), key=lambda x: x[1], reverse=True)[:5]
