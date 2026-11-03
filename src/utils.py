import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Фиксирует случайные состояния для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)


def configure_runtime(num_threads=1):
    """Настраивает количество потоков PyTorch."""
    torch.set_num_threads(num_threads)


def get_versions():
    """Возвращает версии используемых библиотек."""
    import sklearn
    import yaml
    import pandas
    import mlflow
    import optuna

    return {
        "python": os.sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "yaml": yaml.__version__,
        "pandas": pandas.__version__,
        "mlflow": mlflow.__version__,
        "optuna": optuna.__version__
    }
