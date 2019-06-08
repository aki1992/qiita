# -*- coding: utf-8 -*-
import numpy as np
from itertools import combinations_with_replacement


def division_generator(N, K):
    """
    Generate division [d_1, ..., d_K] of N.
    (Each list [d_1, ..., d_K] satisfies that d_1, ..., d_K are non-negative integers
    and the sum of d_1, ..., d_K is equal to N.)
    """
    for c in combinations_with_replacement(np.arange(N - K + 1), K - 1):
        accum = [0] + list(c) + [N - K]
        yield [accum[i + 1] - accum[i] + 1 for i in range(K)]

class TowerOfHawaiTable(object):
    def __init__(self, max_n_poles, max_n_disks):
        self.max_n_poles = max_n_poles
        self.max_n_disks = max_n_disks
        self.table = np.zeros((max_n_poles - 2, max_n_disks), dtype=np.int)
        self.fill_table()

    def fill_table(self):
        for r in range(self.max_n_poles - 2):
            self.fill_row(r)

    def fill_row(self, r):
        if r == 0:
            for i in range(self.max_n_disks):
                self.table[r][i] = 2 ** (i + 1) - 1
            return
        
        for i in range(self.max_n_disks):
            if i < r + 1:
                self.table[r][i] = self.table[r - 1][i]
                continue
            
            min_step = self.table[r - 1][i]
            for d in division_generator(i, r + 1):
                tmp = sum([self.table[j][d[j] - 1] for j in range(r + 1)])
                n_steps = 1 + 2 * tmp
                if n_steps < min_step:
                    min_step = n_steps
            self.table[r][i] = min_step


if __name__ == '__main__':
    table = TowerOfHawaiTable(7, 10)
    print(table.table)