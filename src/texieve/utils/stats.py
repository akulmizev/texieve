import logging
from typing import List

import numpy as np
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


def ks_statistic(sample, full_data):
    return stats.ks_2samp(sample, full_data).statistic


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))


def find_elbow(x, y):
    # Normalize data
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Find the point with maximum distance from the line
    distances = np.abs(
        (y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0]
    ) / np.sqrt((y[-1] - y[0]) ** 2 + (x[-1] - x[0]) ** 2)
    elbow_index = np.argmax(distances)

    return elbow_index


def normalize(data):
    z_scores = stats.zscore(data)
    normalized = (z_scores - np.min(z_scores)) / (np.max(z_scores) - np.min(z_scores))

    return normalized


def get_representative_sample_size(distribution, metric="ks"):
    if metric not in ["ks", "js"]:
        logger.warning(f"Invalid metric '{metric}'. Using 'ks' instead.")
        metric = "ks"

    sample_sizes = np.logspace(1, np.log10(distribution.shape[0]), 25).astype(int)
    sample_sizes = np.unique(sample_sizes)  # Remove duplicates

    # Arrays to store results
    results = []

    # Calculate statistics for each sample size
    for n in tqdm(sample_sizes):
        metrics_n = []
        for _ in range(5):  # Repeat sampling 10 times for robustness
            sample = np.random.choice(distribution, size=n, replace=False)
            if metric == "ks":
                metrics_n.append(ks_statistic(sample, distribution))
            elif metric == "js":
                hist_sample, _ = np.histogram(sample, bins=100, density=True)
                hist_full, _ = np.histogram(distribution, bins=100, density=True)
                metrics_n.append(js_divergence(hist_sample, hist_full))

        results.append(np.mean(metrics_n))

    elbow_index = find_elbow(sample_sizes, results)

    return sample_sizes[elbow_index]


def get_sampling_probs(
    raw_weights: List[float | int], strategy: str = "uniform", temperature: float = 1.0
):
    """
    Calculates sampling probabilities for each language based on the specified strategy.

    Parameters
    ----------
    raw_weights : Tuple[Any]
        The raw weights for each language.
    strategy : str
        The strategy for calculating sampling probabilities.
    temperature : float
        The temperature parameter for temperature-based sampling.

    Returns
    -------
    List[float]
        A list of sampling probabilities for each language.
    """

    if strategy == "uniform":
        probs = [1 / len(raw_weights)] * len(raw_weights)
    elif strategy == "proportional":
        total = sum(raw_weights)
        probs = [weight / total for weight in raw_weights]
    elif strategy == "inverse_proportional":
        inverse_probs = [1 / weight for weight in raw_weights]
        total = sum(inverse_probs)
        probs = [prob / total for prob in inverse_probs]
    elif strategy == "inverse_proportional_sqrt":
        inverse_sqrt_probs = [1 / np.sqrt(weight) for weight in raw_weights]
        total = sum(inverse_sqrt_probs)
        probs = [prob / total for prob in inverse_sqrt_probs]
    elif strategy == "temperature":
        counts = [weight for weight in raw_weights]
        temp_probs = [count ** (1 / temperature) for count in counts]
        total = sum(temp_probs)
        probs = [prob / total for prob in temp_probs]
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    return probs
