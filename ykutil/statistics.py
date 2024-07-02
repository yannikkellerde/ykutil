from collections import defaultdict


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
