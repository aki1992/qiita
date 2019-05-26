# -*- coding: utf-8 -*-

"""
https://qiita.com/akih1992/items/cdb39e5a23dff9b13498
"""

import numpy as np
from collections import defaultdict

global MAX_DISK_SIZE
MAX_DISK_SIZE = 1000

class Pole(list):
    @property
    def top_disk(self):
        if not bool(self):
            return MAX_DISK_SIZE
        else:
            return self[-1]

    def __eq__(self, other):
        return bool(self.top_disk == other.top_disk)
    
    def __gt__(self, other):
        return bool(self.top_disk > other.top_disk)

    def __lt__(self, other):
        return bool(self.top_disk < other.top_disk)

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __le__(self, other):
        return not self.__gt__(other)
    
    def __ge__(self, other):
        return not self.__lt__(other)


class TowerOfHanoiEnvironment(object):
    def __init__(self, n_disks, max_episode_steps=200):
        self.n_disks = n_disks
        self.n_actions = 3
        self.max_episode_steps = max_episode_steps

    def reset(self):
        self.pole = [Pole() for i in range(3)]
        for d in reversed(range(self.n_disks)):
            self.pole[0].append(d)
        self.curr_step = 0
        return self.state
    
    def step(self, action):
        self.curr_step += 1
        if action == 0:
            self.move_disk(0, 1)
        elif action == 1:
            self.move_disk(1, 2)
        elif action == 2:
            self.move_disk(2, 0)

        is_terminal = False
        reward = -1
        
        if (len(self.pole[1]) == self.n_disks) or (len(self.pole[2]) == self.n_disks):
            is_terminal = True
            reward = 1
        elif self.curr_step == self.max_episode_steps:
            is_terminal = True
        
        return self.state, reward, is_terminal

    @property
    def state(self):
        state = []
        for i in range(3):
            state += [bool(j in self.pole[i]) for j in range(self.n_disks)]
        return np.array(state, dtype=np.float32)

    def move_disk(self, a, b):
        if self.pole[a] > self.pole[b]:
            self.pole[a].append(self.pole[b].pop())
        elif self.pole[a] < self.pole[b]:
            self.pole[b].append(self.pole[a].pop())
         
    def render(self):
        print('pole0:{}'.format(self.pole[0]))
        print('pole1:{}'.format(self.pole[1]))
        print('pole2:{}'.format(self.pole[2]))


class QLearning(object):
    """
    Params:
        alpha : learning rate
        gamma : discount rate
    """
    def __init__(self, env, actor, alpha=0.01, gamma=0.99):
        self.env = env
        self.actor = actor
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(lambda: [0 for _ in range(self.env.n_actions)])
        self.training_episode_count = 0


    def play_episode(self, train=True, display=False):
        if train:
            self.training_episode_count += 1
        state = self.env.reset()
        is_terminal = False
        while not is_terminal:
            q_values = self.q_table[tuple(state)]
            if train:
                action = self.actor.act_with_exploration(q_values)
                next_state, reward, is_terminal = self.env.step(action)
                self.update(state, action, reward, is_terminal, next_state)
            else:
                action = self.actor.act_without_exploration(q_values)
                next_state, reward, is_terminal = self.env.step(action)
            if display:
                print('----')
                print('step:{}'.format(self.env.curr_step))
                self.env.render()
            state = next_state

    def update(self, state, action, reward, is_terminal, next_state):
        target = reward + (1 - is_terminal) * max(self.q_table[tuple(next_state)])
        self.q_table[tuple(state)][action] *= self.alpha
        self.q_table[tuple(state)][action] += (1 - self.alpha) * target


class EpsilonGreedyActor(object):
    def __init__(self, epsilon=0.1, random_state=0):
        self.epsilon = epsilon
        self.random = np.random
        self.random.seed(random_state)

    def act_without_exploration(self, q_values):
        max_q = max(q_values)
        argmax_list = [
            action for action, q in enumerate(q_values)
            if q == max_q
        ]
        return self.random.choice(argmax_list)

    def act_with_exploration(self, q_values):
        if self.random.uniform(0, 1) < self.epsilon:
            actions = np.arange(len(q_values))
            return self.random.choice(actions)
        else:
            return self.act_without_exploration(q_values)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    env = TowerOfHanoiEnvironment(n_disks=5, max_episode_steps=200)
    actor = EpsilonGreedyActor(random_state=0)
    model = QLearning(env, actor)
    n_episodes = 200
    episode_steps_traj = []

    print('---- Start Training ----')
    for e in range(n_episodes):
        model.play_episode()
        episode_steps_traj.append(env.curr_step)
        if (e + 1) % 10 == 0:
            print('episode:{} episode_steps:{}'.format(
                model.training_episode_count,
                env.curr_step
            ))
    print('---- Finish Training ----')

    plt.plot(np.arange(n_episodes) + 1, episode_steps_traj, label='learning')
    plt.plot([1, n_episodes + 1], [2**5-1, 2**5-1], label='shortest')
    plt.xlabel('episode')
    plt.ylabel('episode steps')
    plt.legend()
    plt.show()

    
