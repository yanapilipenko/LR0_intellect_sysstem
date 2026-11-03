import argparse
import csv
import hashlib
import json
import subprocess
import time
import traceback
import uuid

from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Позволяет запускать файл непосредственно через python src/train.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import DigitsClassifier
from src.utils import set_seed, configure_runtime, get_versions


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Путь к YAML-конфигурации"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Переопределение seed из конфигурации"
    )

    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Выполнить оценку на тестовой выборке"
    )

    return parser.parse_args()


def load_config(path, seed_override=None):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if seed_override is not None:
        config["seed"] = seed_override

    # Проверка корректности параметров
    assert config["train"]["lr"] > 0, "Learning rate должен быть положительным"
    assert config["train"]["epochs"] == 30, "Количество эпох должно быть равно 30"
    assert config["model"]["n_layers"] >= 1
    assert config["model"]["hidden"] >= 1
    assert 0 <= config["model"]["dropout"] < 1
    assert config["train"]["batch_size"] > 0
    assert config["train"]["weight_decay"] >= 0

    return config


def get_git_info():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True
    ).stdout.strip()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True
    ).stdout.strip()

    return commit, status


def load_data():
    digits = load_digits()

    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    with open("data/split.json", "r", encoding="utf-8") as f:
        split = json.load(f)

    train_idx = np.array(split["train"])
    val_idx = np.array(split["val"])
    test_idx = np.array(split["test"])

    return X, y, train_idx, val_idx, test_idx


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += len(y)

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def get_predictions(model, X, y, indices, split_name, device):
    model.eval()

    x = torch.tensor(X[indices], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        predictions = logits.argmax(dim=1).cpu().numpy()

    return pd.DataFrame({
        "split": split_name,
        "index": indices,
        "y_true": y[indices],
        "y_pred": predictions
    })


def main():
    args = parse_args()

    config = load_config(args.config, args.seed)

    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )

    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()

    commit, git_status = get_git_info()

    # Тестовая оценка допускается только на зафиксированном коде
    if args.eval_test and git_status:
        raise RuntimeError(
            "Перед тестовой оценкой необходимо сохранить все изменения кода в Git."
        )

    metadata = {
        "run_id": run_id,
        "series": config["experiment"],
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "commit": commit,
        "seed": config["seed"],
        "versions": get_versions(),
        "num_threads": config["runtime"]["num_threads"],
        "split_sha256": hashlib.sha256(
            Path("data/split.json").read_bytes()
        ).hexdigest()
    }

    mlflow_run_started = False

    try:
        # Сохраняем конфигурацию запуска
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        # Настройка воспроизводимости
        configure_runtime(config["runtime"]["num_threads"])
        set_seed(config["seed"])

        device = torch.device(config["runtime"]["device"])

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна на этом устройстве")

        # Загружаем данные и индексы разбиения
        X, y, train_idx, val_idx, test_idx = load_data()

        train_dataset = TensorDataset(
            torch.tensor(X[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.long)
        )

        val_dataset = TensorDataset(
            torch.tensor(X[val_idx], dtype=torch.float32),
            torch.tensor(y[val_idx], dtype=torch.long)
        )

        generator = torch.Generator()
        generator.manual_seed(config["seed"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            generator=generator,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=0
        )

        # Создаём модель
        model = DigitsClassifier(
            n_layers=config["model"]["n_layers"],
            hidden=config["model"]["hidden"],
            dropout=config["model"]["dropout"]
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"]
        )

        # Настраиваем MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(config["experiment"])

        mlflow.start_run(run_name=run_id)
        mlflow_run_started = True

        mlflow.log_params({
            "seed": config["seed"],
            "n_layers": config["model"]["n_layers"],
            "hidden": config["model"]["hidden"],
            "dropout": config["model"]["dropout"],
            "lr": config["train"]["lr"],
            "weight_decay": config["train"]["weight_decay"],
            "batch_size": config["train"]["batch_size"],
            "epochs": config["train"]["epochs"]
        })

        mlflow.set_tags({
            "commit": commit,
            "series": config["experiment"],
            "seed": str(config["seed"])
        })

        # Обучение
        history_path = run_dir / "history.csv"
        total_train_time = 0.0

        for epoch in range(1, config["train"]["epochs"] + 1):
            epoch_start = time.perf_counter()

            model.train()

            total_loss = 0.0
            total_samples = 0

            for x, labels in train_loader:
                x = x.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model(x)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

            train_loss = total_loss / total_samples

            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device
            )

            epoch_time = time.perf_counter() - epoch_start
            total_train_time += epoch_time

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time
            }

            # Записываем историю после каждой эпохи
            with open(
                history_path,
                "a",
                newline="",
                encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())

                if epoch == 1:
                    writer.writeheader()

                writer.writerow(row)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time
            }, step=epoch)

            print(
                f"Epoch {epoch:02d}/30 | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

        # Получаем предсказания на валидационной выборке
        predictions = get_predictions(
            model, X, y, val_idx, "val", device
        )

        # Тестовая выборка используется только по запросу
        if args.eval_test:
            test_predictions = get_predictions(
                model, X, y, test_idx, "test", device
            )

            predictions = pd.concat(
                [predictions, test_predictions],
                ignore_index=True
            )

        predictions.to_csv(
            run_dir / "predictions.csv",
            index=False
        )

        # Метрики вычисляются по сохранённым предсказаниям
        val_predictions = predictions[
            predictions["split"] == "val"
        ]

        val_acc = (
            val_predictions["y_true"] == val_predictions["y_pred"]
        ).mean()

        history = pd.read_csv(history_path)

        metrics = {
            "val_acc": float(val_acc),
            "val_loss": float(history.iloc[-1]["val_loss"]),
            "train_time_sec": float(total_train_time)
        }

        if args.eval_test:
            test_predictions = predictions[
                predictions["split"] == "test"
            ]

            test_acc = (
                test_predictions["y_true"]
                == test_predictions["y_pred"]
            ).mean()

            metrics["test_acc"] = float(test_acc)

        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        mlflow.log_metrics(metrics)

        metadata["status"] = "completed"
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()

        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        mlflow.log_artifacts(str(run_dir))

        print("\nОбучение завершено!")
        print("Run ID:", run_id)
        print("Метрики:", metrics)
        print("Результаты сохранены в:", run_dir)

    except Exception:
        metadata["status"] = "failed"
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["error"] = traceback.format_exc()

        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        with open(run_dir / "traceback.txt", "w", encoding="utf-8") as f:
            f.write(metadata["error"])

        raise

    finally:
        if mlflow_run_started:
            mlflow.end_run()


if __name__ == "__main__":
    main()
