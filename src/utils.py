import csv
import os
from rich.console import Console

console = Console()


def log_experiment(results_path, row_dict):
    file_exists = os.path.exists(results_path)

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)


def print_epoch(epoch, train_loss, val_loss):
    console.print(
        f"[bold blue]Epoch {epoch}[/bold blue] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )