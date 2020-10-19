#!/usr/bin/env python3

import json
import collections
import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, window_size) :
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


def plot_data(file_name, sliding_window):
    with open(file_name) as fp:
        data_dict = json.load(fp)

    rewards = []

    episodes = sorted(map(int, list(data_dict.keys())))
    print('expt: {}, episodes: {}'.format(file_name, len(episodes)))

    for ep in episodes:
        rewards.append(sum(data_dict[str(ep)]))

    plt.title('Rewards per Episode')
    plt.plot(moving_average(episodes, sliding_window), moving_average(rewards, sliding_window), label=file_name.split('.')[0])
    plt.legend(loc='best')
    plt.ylabel('rewards')
    plt.xlabel('episodes')


if __name__ == '__main__':
    plt.figure(figsize=(20, 15))
    plot_data('/home/mihir/Documents/Cluster/reward_logs/euc_dist_unscaled.json', sliding_window=10)
    plot_data('/home/mihir/Documents/Cluster/reward_logs/euc_dist_unsc_pen.json', 10)
    plot_data('/home/mihir/Documents/Cluster/reward_logs/paper_dist_sc_pen.json', 10)
    plot_data('/home/mihir/Documents/Cluster/reward_logs/paper_dist_scaled.json', 10)
    plot_data('/home/mihir/Documents/Cluster/reward_logs/ppo_unsc_pen_paper.json', 10)
    plot_data('/home/mihir/Documents/Cluster/reward_logs/sp_ppo_unsc_pen.json', 10)
    plt.grid()
    plt.show()
