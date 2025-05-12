import fire
from core.stream_action_recognizer import StreamActionRecognizer
from core.action_validator import ActionValidator
from loguru import logger

def run_demo(config, checkpoint, labels, camera_url, device='cuda:0', post_url=None,
             window_size=10, threshold=0.8, repetitions=2, id_user=0, resize_width=600):

    logger.info("🎬 Iniciando StreamActionRecognizer...")

    validator = ActionValidator(
        window_size=window_size,
        threshold=threshold,
        repetitions=repetitions,
        post_url=post_url,
        id_user=id_user
    )

    app = StreamActionRecognizer(
        config_path=config,
        checkpoint_path=checkpoint,
        label_path=labels,
        device=device,
        camera_url=camera_url,
        validator=validator,
        resize_width=resize_width
    )

    app.start()

if __name__ == "__main__":
    fire.Fire(run_demo)
