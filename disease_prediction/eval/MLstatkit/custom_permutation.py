# MLstatkit/custom_permutation.py
from typing import Tuple
import numpy as np
from .metrics import get_metric_fn

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def Permutation_test_new(
    y_true,
    prob_model_A,
    prob_model_B,
    metric_str: str = "f1",
    n_bootstraps: int = 1000,
    threshold_A: float = 0.5,
    threshold_B: float = 0.5,
    average: str = "binary",
    random_state: int = 0
) -> Tuple[float, float, float, float, float, float]:
    """
    Return: metric_a, metric_b, p_value, benchmark, samples_mean, samples_std
    """
    y_true = np.asarray(y_true)
    a = np.asarray(prob_model_A)
    b = np.asarray(prob_model_B)
    assert y_true.shape[0] == a.shape[0] == b.shape[0]

    rng = np.random.RandomState(random_state)
    metric_fn_a = get_metric_fn(metric_str, threshold_A, average)
    metric_fn_b = get_metric_fn(metric_str, threshold_B, average)

    metric_a = metric_fn_a(y_true, a)
    metric_b = metric_fn_b(y_true, b)
    benchmark = abs(metric_a - metric_b)

    samples = np.zeros(n_bootstraps, dtype=float)
    for i in tqdm(range(n_bootstraps), desc=f"Computing {metric_str} Permutation Test p-value"):
        msk = rng.rand(len(y_true)) < 0.5
        a_perm = np.where(msk, a, b)
        b_perm = np.where(msk, b, a)
        metric_a_perm = metric_fn_a(y_true, a_perm)
        metric_b_perm = metric_fn_b(y_true, b_perm)
        samples[i] = abs(metric_a_perm - metric_b_perm)

    p_value = float(np.mean(samples >= benchmark))
    return float(metric_a), float(metric_b), p_value, float(benchmark), float(samples.mean()), float(samples.std(ddof=0))
