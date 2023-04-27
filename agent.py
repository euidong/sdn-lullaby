import numpy as np

from dataType import Server


class Agent:
    MAX_HISOTRY_LEN = 1000

    def __init__(self, action_num: int, state_num: int, epsilon: float = 0.5, alpha=0.1, gamma=0.1) -> None:
        self.action_num = action_num
        self.state_num = state_num
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.history = []
        self.q_table = np.random.rand(state_num, action_num)

    def decide_action(self, state, duration) -> int:
        if np.random.uniform() < self.epsilon / duration:
            action = np.random.choice(self.action_num)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update(self, state, action, reward, next_state) -> None:
        if len(self.history) >= self.MAX_HISOTRY_LEN:
            self.history.pop(0)
        self.history.append((state, action, reward, next_state))
        if next_state is None:
            self.q_table[state][action] += self.alpha * \
                (reward - self.q_table[state][action])
        else:
            self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(
                self.q_table[next_state]) - self.q_table[state][action])

    # save q_table
    def save(self) -> None:
        np.save("q_table.npy", self.q_table)

    # load q_table
    def load(self) -> None:
        self.q_table = np.load("q_table.npy")
