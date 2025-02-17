import time
from collections import defaultdict
from math import sqrt
from scipy.stats import beta


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


def clopper_pearson_interval(successes, trials, confidence_level=0.95):
    """
    Compute the Clopper-Pearson confidence interval.

    Parameters:
        successes (int): Number of successes.
        trials (int): Total number of trials.
        confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence).

    Returns:
        (lower_bound, upper_bound): The confidence interval bounds.
    """
    alpha = 1 - confidence_level
    lower_bound = (
        beta.ppf(alpha / 2, successes, trials - successes + 1) if successes > 0 else 0.0
    )
    upper_bound = (
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
        if successes < trials
        else 1.0
    )
    return lower_bound, upper_bound
