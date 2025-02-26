import time
from collections import defaultdict
from math import sqrt
import numpy as np


class Statlogger:
    def __init__(self):
        self.statistics = defaultdict(float)
        self.sum_stats = defaultdict(float)
        self.statnumbers = defaultdict(int)
        self.timers = dict()

    def reset(self):
        self.statistics.clear()
        self.sum_stats.clear()
        self.statnumbers.clear()

    def update(self, key, value):
        self.statistics[key] = (
            (self.statistics[key] * self.statnumbers[key]) + value
        ) / (self.statnumbers[key] + 1)
        self.statnumbers[key] = self.statnumbers[key] + 1

    def update_sum_stat(self, key, value):
        self.sum_stats[key] += value

    def start_timer(self, key):
        self.timers[key] = time.perf_counter()

    def stop_timer(self, key, average=False, summed=True):
        assert summed or average
        if average:
            self.update(f"average_{key}", time.perf_counter() - self.timers[key])
        if summed:
            self.update_sum_stat(f"total_{key}", time.perf_counter() - self.timers[key])
        del self.timers[key]

    @property
    def stats(self):
        some_dic = self.statistics.copy()
        some_dic.update(self.sum_stats)
        return some_dic


class Welfords:
    def __init__(self):
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        return sqrt(self.M2 / self.count) if self.count > 1 else 0.0

    def normalize(self, value):
        return (value - self.mean) / self.std if self.std > 0 else 0.0

    def unnormalize(self, value):
        return value * self.std + self.mean

    def __repr__(self):
        return f"Welfords(Mean: {self.mean}, Std: {self.std}, Count: {self.count})"


def compute_metric(pred: float, targ: float, metric_type: str):
    pred = float(pred)
    targ = float(targ)
    match metric_type:
        case "mae":
            return abs(pred - targ)
        case "mse":
            return (pred - targ) ** 2
        case "acc":
            return int(round(pred) == targ)
        case "bce":
            return -targ * np.log(pred) - (1 - targ) * np.log(1 - pred)
        case _:
            raise ValueError(f"Unknown metric type {metric_type}")


def monte_carlo_bernoulli_p_value(s1: int, t1: int, s2: int, t2: int, n: int = 10000):
    """Given samples of two bernoulli distributions, compute the monte-carlo p-value.
    Bayesian approach using sampling

    >>> np.random.seed(42)
    >>> 0.001 < monte_carlo_bernoulli_p_value(70, 100, 50, 100) < 0.003
    True
    """
    # Beta values for Bayesian estimation
    f1 = t1 - s1 + 1
    f2 = t2 - s2 + 1
    s1 = s1 + 1
    s2 = s2 + 1

    # Draw samples from the Beta distributions
    samples_p1 = np.random.beta(s1, f1, n)
    samples_p2 = np.random.beta(s2, f2, n)

    # Compute probability that p1 > p2
    prob_p1_greater_p2 = np.mean(samples_p1 > samples_p2)

    return 1 - prob_p1_greater_p2


if __name__ == "__main__":
    import doctest

    doctest.testmod()
