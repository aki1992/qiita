# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from itertools import combinations


class TowerOfHanoiEnvironment(object):
    def __init__(self, n_disks, n_poles=3,max_episode_steps=200):
        self.n_disks = n_disks
        self.n_poles = n_poles

        self.action_map = [
            c for c in combinations(np.arange(self.n_poles), 2)
        ]
        self.n_actions = len(self.action_map)
        self.max_episode_steps = max_episode_steps
        
        state = np.zeros((self.n_poles * self.n_disks),dtype=np.float32)
        state[:self.n_disks].fill(1)
        self.state = state
        self.curr_step = 0

    def reset(self):
        self.state.fill(0)
        self.state[:self.n_disks].fill(1)
        self.curr_step = 0
        return self.state
    
    def step(self, action):
        self.curr_step += 1
        result = self.move_disk(*self.action_map[action])

        is_terminal = False
        reward = -0.1
        
        if self.curr_step >= self.max_episode_steps:
            is_terminal = True
        
        #No move
        if result == 0:
            reward = -1

        #Achivement
        if not np.any(self._pole_state == 0):
            for pole_id in range(1, self.n_poles):
                if np.all(self._pole_state(pole_id) == 1):
                    is_terminal = True
                    reward = 1
                    break

        return self.state.copy(), reward, is_terminal

    def _top_disk(self, pole_id):
        """
        If the given pole is empty, then return n_disk which is bigger than
        any disk id.
        """
        for disk_id, val in enumerate(self._pole_state(pole_id)):
            if val == 1:
                return disk_id
        return self.n_disks 

    def _pole_state(self, pole_id):
        start = pole_id * self.n_disks
        return self.state[start : start + self.n_disks]

    def move_disk(self, p1, p2):
        """
        If a disk move, return 1. Otherwise, return 0.
        """
        top_p1 = self._top_disk(p1)
        top_p2 = self._top_disk(p2)
        
        #No move
        if top_p1 == top_p2:
            return 0
        
        if top_p1 > top_p2:
            self._pole_state(p1)[top_p2] = 1
            self._pole_state(p2)[top_p2] = 0
        elif top_p1 < top_p2:
            self._pole_state(p2)[top_p1] = 1
            self._pole_state(p1)[top_p1] = 0
        return 1
         
    def render(self):
        for pole_id in range(self.n_poles):
            disks = np.where(self._pole_state(pole_id))[0]
            print(f'pole{pole_id}: {list(reversed(disks))}')


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
                print(f'step:{self.env.curr_step}')
                self.env.render()
            state = next_state
        
    def update(self, state, action, reward, is_terminal, next_state):
        target = reward + (1 - is_terminal) * max(self.q_table[tuple(next_state)])
        self.q_table[tuple(state)][action] *= self.alpha
        self.q_table[tuple(state)][action] += (1 - self.alpha) * target


class EpsilonGreedyActor(object):
    def __init__(self, epsilon=0.1, random_state=None):
        self.epsilon = epsilon
        self.random = np.random
        if random_state is not None:
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
    import datetime
    import matplotlib.pyplot as plt
    
    env = TowerOfHanoiEnvironment(n_disks=8, n_poles=4, max_episode_steps=2000)
    actor = EpsilonGreedyActor(random_state=0)
    model = QLearning(env, actor, alpha=0.1)
    episode_steps_traj = []

    print(datetime.datetime.now())
    print('---- Start Training ----')
    for e in range(10000):
        model.play_episode()
        episode_steps_traj.append(env.curr_step)
        if (e + 1) % 100 == 0:
            print('episode:{} episode_steps(mean):{}'.format(
                model.training_episode_count,
                np.mean(episode_steps_traj[-100:])
                
            ))
    print('---- Finish Training ----')
    print(datetime.datetime.now())

    mean_traj = [np.mean(episode_steps_traj[max(0, i - 99):i + 1]) for i in range(10000)]
    plt.plot(np.arange(1, 10001), episode_steps_traj, label='observed')
    plt.plot(np.arange(1, 10001), mean_traj, label='mean')
    plt.plot([1, 10001], [33, 33], label='shortest')
    plt.xlabel('episode')
    plt.ylabel('episode steps')
    plt.ylim(0, 2100)
    plt.legend()
    plt.show()

