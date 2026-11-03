import argparse
import copy
import csv
import hashlib
import json
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import optuna
import pandas as pd
import torch
import yaml

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import load_data, evaluate, get_predictions, get_git_info
from src.model import DigitsClassifier
from src.utils import set_seed, configure_runtime, get_versions


TOTAL_TRIALS = 40
STUDY_NAME = "lr0_search"


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-new-trials",
        type=int,
        default=None,
        help="Максимальное количество новых испытаний"
    )

    return parser.parse_args()


def objective(trial, base_config, data):
    # Подбираем гиперпараметры
    config = copy.deepcopy(base_config)

    config["model"]["n_layers"] = trial.suggest_int(
        "n_layers", 1, 3
    )

    config["model"]["hidden"] = trial.suggest_int(
        "hidden", 32, 512, log=True
    )

    config["model"]["dropout"] = trial.suggest_float(
        "dropout", 0.0, 0.5
    )

    config["train"]["lr"] = trial.suggest_float(
        "lr", 1e-4, 1e-1, log=True
    )

    config["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 1e-2, log=True
    )

    config["train"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128]
    )

    seed = config["seed"]

    # Создаём папку отдельного запуска
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_trial"
        + str(trial.number)
        + "_"
        + uuid.uuid4().hex[:6]
    )

    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    commit, _ = get_git_info()

    metadata = {
        "run_id": run_id,
        "series": STUDY_NAME,
        "trial_number": trial.number,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "commit": commit,
        "seed": seed,
        "versions": get_versions(),
        "num_threads": config["runtime"]["num_threads"],
        "split_sha256": hashlib.sha256(
            Path("data/split.json").read_bytes()
        ).hexdigest()
    }

    history_path = run_dir / "history.csv"
    total_train_time = 0.0
    last_val_acc = None
    last_val_loss = None

    with mlflow.start_run(
        run_name=f"trial_{trial.number}",
        nested=True
    ) as mlrun:

        trial.set_user_attr(
            "mlflow_run_id", mlrun.info.run_id
        )

        mlflow.log_params({
            "seed": seed,
            "n_layers": config["model"]["n_layers"],
            "hidden": config["model"]["hidden"],
            "dropout": config["model"]["dropout"],
            "lr": config["train"]["lr"],
            "weight_decay": config["train"]["weight_decay"],
            "batch_size": config["train"]["batch_size"],
            "epochs": config["train"]["epochs"]
        })

        mlflow.set_tags({
            "series": STUDY_NAME,
            "trial_number": str(trial.number),
            "commit": commit
        })

        try:
            save_json(run_dir / "metadata.json", metadata)

            with open(
                run_dir / "config.yaml",
                "w",
                encoding="utf-8"
            ) as f:
                yaml.safe_dump(
                    config, f,
                    allow_unicode=True,
                    sort_keys=False
                )

            # Воспроизводимость каждого испытания
            configure_runtime(config["runtime"]["num_threads"])
            set_seed(seed)

            device = torch.device(config["runtime"]["device"])

            X, y, train_idx, val_idx, test_idx = data

            train_dataset = TensorDataset(
                torch.tensor(X[train_idx], dtype=torch.float32),
                torch.tensor(y[train_idx], dtype=torch.long)
            )

            val_dataset = TensorDataset(
                torch.tensor(X[val_idx], dtype=torch.float32),
                torch.tensor(y[val_idx], dtype=torch.long)
            )

            generator = torch.Generator()
            generator.manual_seed(seed)

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

            # Обучение модели
            for epoch in range(1, 31):
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
                    model, val_loader, criterion, device
                )

                epoch_time = time.perf_counter() - epoch_start
                total_train_time += epoch_time

                last_val_acc = val_acc
                last_val_loss = val_loss

                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time_sec": epoch_time
                }

                with open(
                    history_path,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as f:
                    writer = csv.DictWriter(
                        f, fieldnames=row.keys()
                    )

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
                    f"Trial {trial.number:02d} | "
                    f"Epoch {epoch:02d}/30 | "
                    f"val_acc={val_acc:.4f}"
                )

                # Передаём точность Optuna для Pruner
                trial.report(val_acc, step=epoch)

                if trial.should_prune():
                    metadata["status"] = "pruned"
                    metadata["finished_at"] = (
                        datetime.now(timezone.utc).isoformat()
                    )

                    save_json(
                        run_dir / "metrics.json",
                        {
                            "val_acc": float(val_acc),
                            "val_loss": float(val_loss),
                            "train_time_sec": float(total_train_time)
                        }
                    )

                    save_json(run_dir / "metadata.json", metadata)

                    mlflow.set_tag("trial_state", "PRUNED")
                    mlflow.log_metric("final_val_acc", val_acc)
                    mlflow.log_artifacts(str(run_dir))

                    raise optuna.TrialPruned()

            # После 30 эпох сохраняем предсказания
            predictions = get_predictions(
                model, X, y, val_idx, "val", device
            )

            predictions.to_csv(
                run_dir / "predictions.csv",
                index=False
            )

            val_acc_from_predictions = (
                predictions["y_true"] == predictions["y_pred"]
            ).mean()

            metrics = {
                "val_acc": float(val_acc_from_predictions),
                "val_loss": float(last_val_loss),
                "train_time_sec": float(total_train_time)
            }

            save_json(run_dir / "metrics.json", metrics)

            metadata["status"] = "completed"
            metadata["finished_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

            save_json(run_dir / "metadata.json", metadata)

            mlflow.log_metrics(metrics)
            mlflow.set_tag("trial_state", "COMPLETE")
            mlflow.log_artifacts(str(run_dir))

            return metrics["val_acc"]

        except optuna.TrialPruned:
            raise

        except Exception:
            metadata["status"] = "failed"
            metadata["finished_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

            import traceback

            metadata["error"] = traceback.format_exc()

            save_json(run_dir / "metadata.json", metadata)

            with open(
                run_dir / "traceback.txt",
                "w",
                encoding="utf-8"
            ) as f:
                f.write(metadata["error"])

            mlflow.set_tag("trial_state", "FAILED")
            mlflow.log_artifacts(str(run_dir))

            raise


def main():
    args = parse_args()

    with open("configs/search.yaml", "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    base_config["experiment"] = STUDY_NAME
    base_config["seed"] = 0
    base_config["train"]["epochs"] = 30

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(STUDY_NAME)

    sampler = optuna.samplers.TPESampler(seed=42)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage="sqlite:///optuna.db",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    # Считаем уже существующие испытания
    completed_count = len(study.trials)
    remaining = max(0, TOTAL_TRIALS - completed_count)

    if args.max_new_trials is not None:
        if args.max_new_trials < 1:
            raise ValueError("--max-new-trials должен быть положительным")
        trials_to_run = min(remaining, args.max_new_trials)
    else:
        trials_to_run = remaining

    print(f"Уже сохранено испытаний: {completed_count}")
    print(f"Осталось до 40: {remaining}")
    print(f"Будет запущено сейчас: {trials_to_run}")

    if trials_to_run == 0:
        print("Все 40 испытаний уже выполнены.")
        return

    data = load_data()

    try:
        with mlflow.start_run(run_name="optuna_search"):
            mlflow.set_tags({
                "series": STUDY_NAME,
                "sampler": "TPESampler",
                "pruner": "MedianPruner"
            })

            study.optimize(
                lambda trial: objective(
                    trial, base_config, data
                ),
                n_trials=trials_to_run,
                gc_after_trial=True,
                show_progress_bar=True
            )

    except KeyboardInterrupt:
        print("\nПоиск прерван. Состояние испытаний сохранено в optuna.db.")

    print("\nПоиск завершён или приостановлен.")
    print("Всего испытаний:", len(study.trials))

    print("\nЛучший результат:")
    if study.best_trial is not None:
        print("Trial:", study.best_trial.number)
        print("Val accuracy:", study.best_value)
        print("Параметры:", study.best_params)


if __name__ == "__main__":
    main()
