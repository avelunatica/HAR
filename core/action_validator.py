import csv
import os
import requests
from loguru import logger

class ActionValidator:
    def __init__(self, window_size=10, threshold=0.8, repetitions=2, post_url=None, id_user=0):
        self.window_size = window_size
        self.threshold = threshold
        self.repetitions = repetitions
        self.post_url = post_url
        self.id_user = id_user
        self.history = []

        os.makedirs("data", exist_ok=True)
        if not os.path.exists("data/acciones.csv"):
            with open("data/acciones.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["action", "percentage", "correct"])

    def update(self, results):
        top_label, top_score = results[0]

        if top_score >= self.threshold:
            self.history.append(top_label)
            if self.history.count(top_label) >= self.repetitions:
                self.history.clear()
                logger.success(f"Acción validada: {top_label} ({top_score*100:.2f}%)")

                if self.post_url:
                    try:
                        response = requests.post(self.post_url, json={
                            "id_user": self.id_user,
                            "action": top_label,
                            "action_percentage": top_score*100
                        })
                        logger.info(f"POST enviado correctamente ({response.status_code})")
                    except Exception as e:
                        logger.warning(f"Error enviando POST: {e}")

                try:
                    with open("data/acciones.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([top_label, round(top_score * 100, 2), ""])
                        logger.info(f"Acción '{top_label}' guardada en CSV.")
                except Exception as e:
                    logger.warning(f"Error guardando acción en CSV: {e}")

                return top_label
        return None
