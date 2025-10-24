import random
import numpy as np
import math
import pickle
from collections import defaultdict

class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-5

    def add(self, state, action, reward, next_state, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append((state, action, reward, next_state))
        self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], []
        
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        
        if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
            probabilities = np.ones_like(probabilities) / len(probabilities)
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probabilities, replace=False)
        samples = [self.buffer[i] for i in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)
        
        return samples, weights, indices

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            if i < len(self.priorities):
                self.priorities[i] = (abs(td_error) + self.epsilon) ** self.alpha
                
class DuelingDoubleQLearningAgent:
    def __init__(self, player_symbol, epsilon_start=0.99, epsilon_end=0.01, epsilon_decay=70000):
        self.player_symbol = player_symbol
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.alpha_start = 0.3
        self.alpha_end = 0.01
        self.alpha_decay = 70000
        self.alpha = self.alpha_start
        self.gamma = 0.9
        self.total_reward = 0
        self.episode_rewards = []
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.episode = 0
        self.step_count = 0
        self.ucb_c = 2

        self.value_table_1 = defaultdict(float)
        self.advantage_table_1 = defaultdict(lambda: np.zeros(9))
        self.value_table_2 = defaultdict(float)
        self.advantage_table_2 = defaultdict(lambda: np.zeros(9))

        self.experience_replay = PrioritizedExperienceReplay(capacity=100000)
        self.batch_size = 64
        self.action_counts = defaultdict(lambda: np.zeros(9))

    def get_q_values(self, state, table_index):
        state_tuple = tuple(state)
        if table_index == 1:
            return self.value_table_1[state_tuple] + self.advantage_table_1[state_tuple] - np.mean(self.advantage_table_1[state_tuple])
        else:
            return self.value_table_2[state_tuple] + self.advantage_table_2[state_tuple] - np.mean(self.advantage_table_2[state_tuple])

    def choose_action(self, state, available_actions, game):
        state_tuple = tuple(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions), "Explorative"
        
        winning_move = game.get_winning_move(self.player_symbol)
        if winning_move is not None and winning_move in available_actions:
            return winning_move, "Attempting to win"
        
        blocking_move = game.get_blocking_move(self.player_symbol)
        if blocking_move is not None and blocking_move in available_actions:
            return blocking_move, "Defending"
        
        total_count = sum(self.action_counts[state_tuple])
        ucb_values = []
        q_values = self.get_q_values(state, 1) if random.random() < 0.5 else self.get_q_values(state, 2)
        for action in available_actions:
            q_value = q_values[action]
            count = self.action_counts[state_tuple][action]
            if count == 0:
                ucb_values.append(float('inf'))
            else:
                ucb_value = q_value + self.ucb_c * math.sqrt(math.log(total_count + 1) / count)
                ucb_values.append(ucb_value)
        
        max_ucb = max(ucb_values)
        best_actions = [a for a, v in zip(available_actions, ucb_values) if v == max_ucb]
        chosen_action = random.choice(best_actions)
        
        self.action_counts[state_tuple][chosen_action] += 1
        return chosen_action, "Attacking"

    def update_q_table(self, state, action, reward, next_state):
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)
    
        if random.random() < 0.5:
            update_value, update_advantage = self.value_table_1, self.advantage_table_1
            target_q_values = self.get_q_values(next_state, 2)
        else:
            update_value, update_advantage = self.value_table_2, self.advantage_table_2
            target_q_values = self.get_q_values(next_state, 1)
    
        q_next = np.max(target_q_values) if ' ' in next_state else 0
        target = reward + self.gamma * q_next
    
        current_q = update_value[state_tuple] + update_advantage[state_tuple][action] - np.mean(update_advantage[state_tuple])
        td_error = target - current_q
    
        update_value[state_tuple] += self.alpha * td_error
        update_advantage[state_tuple][action] += self.alpha * td_error
    
        self.experience_replay.add(state_tuple, action, reward, next_state_tuple, td_error)
    
        if len(self.experience_replay.buffer) >= self.batch_size:
            samples, weights, indices = self.experience_replay.sample(self.batch_size)
            if samples:
                td_errors = []
                for (s, a, r, ns), w in zip(samples, weights):
                    if random.random() < 0.5:
                        q_next = np.max(self.get_q_values(ns, 1))
                        current_q = self.get_q_values(s, 1)[a]
                    else:
                        q_next = np.max(self.get_q_values(ns, 2))
                        current_q = self.get_q_values(s, 2)[a]
                
                    target = r + self.gamma * q_next
                    td_error_sample = target - current_q
                    td_errors.append(td_error_sample)
                
                    if random.random() < 0.5:
                        self.value_table_1[s] += self.alpha * w * td_error_sample
                        self.advantage_table_1[s][a] += self.alpha * w * td_error_sample
                    else:
                        self.value_table_2[s] += self.alpha * w * td_error_sample
                        self.advantage_table_2[s][a] += self.alpha * w * td_error_sample
            
                self.experience_replay.update_priorities(indices, td_errors)
    
        self.total_reward += reward
    
    def reset_reward(self):
        self.total_reward = 0

    def update_episode_reward(self):
        self.episode_rewards.append(self.total_reward)
        self.reset_reward()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_start - self.episode / self.epsilon_decay)

    def update_alpha(self):
        self.alpha = max(self.alpha_end, self.alpha_start - self.episode / self.alpha_decay)

    def update_stats(self, result):
        self.episode += 1
        if result == 'Tie':
            self.ties += 1
        elif result == self.player_symbol:
            self.wins += 1
        else:
            self.losses += 1
        self.update_epsilon()
        self.update_alpha()

    def increase_step_count(self):
        self.step_count += 1

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'value_table_1': dict(self.value_table_1),
                'advantage_table_1': dict(self.advantage_table_1),
                'value_table_2': dict(self.value_table_2),
                'advantage_table_2': dict(self.advantage_table_2),
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'wins': self.wins,
                'losses': self.losses,
                'ties': self.ties,
                'episode': self.episode,
                'step_count': self.step_count,
                'action_counts': dict(self.action_counts)
            }, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.value_table_1 = defaultdict(float, data.get('value_table_1', {}))
            self.advantage_table_1 = defaultdict(lambda: np.zeros(9), data.get('advantage_table_1', {}))
            self.value_table_2 = defaultdict(float, data.get('value_table_2', {}))
            self.advantage_table_2 = defaultdict(lambda: np.zeros(9), data.get('advantage_table_2', {}))
            self.epsilon = data.get('epsilon', self.epsilon_start)
            self.alpha = data.get('alpha', self.alpha_start)
            self.wins = data.get('wins', 0)
            self.losses = data.get('losses', 0)
            self.ties = data.get('ties', 0)
            self.episode = data.get('episode', 0)
            self.step_count = data.get('step_count', 0)
            self.action_counts = defaultdict(lambda: np.zeros(9), data.get('action_counts', {}))