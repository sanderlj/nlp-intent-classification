"""Create CSV reports from experiment results."""

from pathlib import Path
import csv
from src.experiments import ExperimentResult

def save_results_csv(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "language",
                "model_name",
                "feature_name",
                "dev_accuracy",
                "test_accuracy",
                "hyperparameters",
                "notes",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "language": r.language,
                    "model_name": r.model_name,
                    "feature_name": r.feature_name,
                    "dev_accuracy": f"{r.dev_accuracy:.6f}",
                    "test_accuracy": f"{r.test_accuracy:.6f}",
                    "hyperparameters": str(r.hyperparameters),
                    "notes": r.notes,
                }
            )
