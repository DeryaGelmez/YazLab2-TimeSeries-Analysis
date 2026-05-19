import csv
from pathlib import Path
from datetime import datetime


def log_experiment_result(log_path, experiment_info):

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = log_path.exists()

    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=experiment_info.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment_info)