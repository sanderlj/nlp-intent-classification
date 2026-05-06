"""Create CSV reports from experiment results."""

from pathlib import Path
import csv
import json
import matplotlib.pyplot as plt
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

def fmt_tokens(value):
    if isinstance(value, list):
        return ", ".join(value)
    return value

def save_tokenization_examples(language: str, examples: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{language}_tokenization_examples.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["example_id", "intent", "text", "tokens_k100", "tokens_k300", "tokens_k500"])
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "example_id": example.get("example_id", ""),
                    "intent": example.get("intent", ""),
                    "text": example.get("text", ""),
                    "tokens_k100": fmt_tokens(example.get("tokens_k100", "")),
                    "tokens_k300": fmt_tokens(example.get("tokens_k300", "")),
                    "tokens_k500": fmt_tokens(example.get("tokens_k500", "")),
                }
            )


def save_error_analysis_json(language: str, analysis: dict, output_dir: Path) -> Path:
    """Save one language's error-analysis artifact as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{language}_error_analysis.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    return output_path


def save_part3_alpha_curve_plot(curve_rows: list[dict], output_path: Path) -> Path:
    """Save a dev-accuracy-vs-alpha curve plot for Part 3 (both languages)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_language: dict[str, list[dict]] = {}
    for row in curve_rows:
        language = str(row["language"])
        by_language.setdefault(language, []).append(row)

    plt.figure(figsize=(8, 5))
    for language, rows in by_language.items():
        rows_sorted = sorted(rows, key=lambda r: float(r["alpha"]))
        x_vals = [float(r["alpha"]) for r in rows_sorted]
        y_vals = [float(r["dev_accuracy"]) for r in rows_sorted]
        k_val = rows_sorted[0]["k"] if rows_sorted else "?"
        plt.plot(x_vals, y_vals, marker="o", label=f"{language} (k={k_val})")

    plt.xscale("log")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Dev Accuracy")
    plt.title("Part 3 NB: Dev Accuracy vs Alpha")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
