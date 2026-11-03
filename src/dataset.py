import json
import hashlib
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def prepare_data():
    # Загружаем набор данных
    digits = load_digits()

    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    # Индексы всех объектов
    indices = np.arange(len(y))

    # Сначала выделяем тестовую выборку — 20%
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=0,
        stratify=y
    )

    # Из оставшихся данных выделяем валидационную — 25%
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=0,
        stratify=y[train_val_idx]
    )

    # Проверяем размеры выборок
    assert len(train_idx) == 1077
    assert len(val_idx) == 360
    assert len(test_idx) == 360

    # Проверяем, что выборки не пересекаются
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    # Проверяем, что все объекты распределены
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(y)

    # Проверяем наличие всех классов в каждой выборке
    for part in [train_idx, val_idx, test_idx]:
        assert set(np.unique(y[part])) == set(range(10))

    # Создаём папку data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Сохраняем индексы разбиения
    split = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist()
    }

    split_path = data_dir / "split.json"

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=4)

    # Вычисляем SHA-256 файла разбиения
    split_hash = hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest()

    # Формируем манифест
    manifest = {
        "dataset": "sklearn.datasets.load_digits",
        "sklearn_version": sklearn.__version__,
        "total_samples": len(y),
        "image_shape": list(digits.images.shape[1:]),
        "number_of_features": X.shape[1],
        "classes": sorted(np.unique(y).tolist()),
        "normalization": "X = X / 16.0",
        "split_method": "stratified train_test_split",
        "split_seed": 0,
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        },
        "split_sha256": split_hash
    }

    with open(data_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print("Данные успешно подготовлены!")
    print(f"Всего объектов: {len(y)}")
    print(f"Train: {len(train_idx)}")
    print(f"Validation: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")
    print(f"SHA-256: {split_hash}")


if __name__ == "__main__":
    prepare_data()
