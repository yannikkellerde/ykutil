from collections import defaultdict
from math import sqrt


class Statlogger:
    def __init__(self):
        self.statistics = defaultdict(float)
        self.sum_stats = defaultdict(float)
        self.statnumbers = defaultdict(int)

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
