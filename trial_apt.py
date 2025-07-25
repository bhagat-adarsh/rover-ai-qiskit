# mars_rover_sim.py

import numpy as np
import tensorflow as tf
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
import random

class MarsEnv:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.rover_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        obstacle_x = random.randint(2, self.size - 2)
        obstacle_y = random.randint(2, self.size - 2)
        if [obstacle_x, obstacle_y] != self.rover_pos and [obstacle_x, obstacle_y] != self.goal_pos:
            self.grid[obstacle_x, obstacle_y] = -1
        return self.get_state()

    def get_state(self):
        return np.array(self.rover_pos + self.goal_pos, dtype=np.float32) / self.size

    def step(self, action):
        x, y = self.rover_pos
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = directions[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            if self.grid[new_x, new_y] == -1:
                return self.get_state(), -5, False
            self.rover_pos = [new_x, new_y]

        reward = 10 if self.rover_pos == self.goal_pos else -0.1
        done = self.rover_pos == self.goal_pos
        return self.get_state(), reward, done

def grover_select_action(q_values):
    n = int(np.ceil(np.log2(len(q_values))))
    best_action = np.argmax(q_values)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    for _ in range(1):
        qc.h(range(n))
        qc.x(range(n))
        if n > 1:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))

    qc.measure(range(n), range(n))

    simulator = AerSimulator()
    job = simulator.run(qc, shots=1)
    result = job.result()
    counts = result.get_counts()

    action = int(max(counts, key=counts.get), 2)
    return action if action < len(q_values) else best_action

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_vals = self.model.predict(np.array([state]), verbose=0)[0]
        return grover_select_action(q_vals)

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
        target_f = self.model.predict(np.array([state]), verbose=0)
        target_f[0][action] = target
        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = MarsEnv(size=6)
    agent = DQNAgent(state_size=4, action_size=4)

    for episode in range(50):
        state = env.reset()
        total_reward = 0

        for step in range(50):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        print(f"Episode {episode+1}: Total Reward = {round(total_reward, 2)}")
