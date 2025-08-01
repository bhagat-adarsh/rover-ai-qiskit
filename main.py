import numpy as np
import tensorflow as tf
from qiskit import QuantumCircuit
from qiskit_ibm_provider import IBMProvider
import random
import pygame
import sys
from collections import deque
import math



import os
from dotenv import load_dotenv
from qiskit_ibm_provider import IBMProvider

load_dotenv()
token = os.getenv("IBM_QUANTUM_TOKEN")
provider = IBMProvider(token=token)



# Store best reward
best_total_reward = float('-inf')


class MarsEnv:
    def __init__(self, size=8, num_obstacles=10):
        self.size = size
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.rover_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        placed = 0
        while placed < self.num_obstacles:
            x, y = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
            if [x, y] != self.rover_pos and [x, y] != self.goal_pos and self.grid[x, y] != -1:
                self.grid[x, y] = -1
                placed += 1
        return self.get_state()

    def get_state(self):
        return np.array(self.rover_pos + self.goal_pos, dtype=np.float32) / (self.size - 1)

    def step(self, action):
        x, y = self.rover_pos
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = moves[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            if self.grid[new_x, new_y] == -1:
                return self.get_state(), -5, False
            self.rover_pos = [new_x, new_y]

        done = self.rover_pos == self.goal_pos
        reward = 10 if done else -0.1
        return self.get_state(), reward, done

def grover_select_action(q_values, provider):
    original_len = len(q_values)
    padded_len = 2 ** math.ceil(math.log2(original_len))
    padded_q = np.zeros(padded_len)
    padded_q[:original_len] = q_values
    best_action = int(np.argmax(padded_q))
    n = int(math.log2(padded_len))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    # Oracle for best action
    bin_str = format(best_action, f'0{n}b')
    for i, bit in enumerate(bin_str):
        if bit == '0':
            qc.x(i)
    if n > 1:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    else:
        qc.z(0)
    for i, bit in enumerate(bin_str):
        if bit == '0':
            qc.x(i)

    # Diffusion operator
    qc.h(range(n))
    qc.x(range(n))
    if n > 1:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))

    qc.measure(range(n), range(n))

    backend = provider.get_backend("ibm_mumbai")
    job = backend.run(qc, shots=1)
    result = job.result()
    counts = result.get_counts()
    action = int(max(counts, key=counts.get), 2)

    return action if action < original_len else best_action

class DQNAgent:
    def __init__(self, state_size, action_size, provider):
        self.state_size = state_size
        self.action_size = action_size
        self.provider = provider
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_vals = self.model.predict(np.array([state]), verbose=0)[0]
        return grover_select_action(q_vals, self.provider)

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for s, a, r, s_, done in minibatch:
            target = r
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(np.array([s_]), verbose=0)[0])
            target_f = self.model.predict(np.array([s]), verbose=0)
            target_f[0][a] = target
            self.model.fit(np.array([s]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def visualize(rover_pos, goal_pos, grid_size, grid):
    pygame.init()
    cell_size = 500 // grid_size
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Mars Rover Simulation")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((255, 255, 255))

    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            color = (0, 0, 0) if grid[x][y] == -1 else (255, 255, 255)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

    pygame.draw.circle(screen, (0, 0, 255), (rover_pos[1] * cell_size + cell_size // 2, rover_pos[0] * cell_size + cell_size // 2), cell_size // 3)
    pygame.draw.circle(screen, (0, 255, 0), (goal_pos[1] * cell_size + cell_size // 2, goal_pos[0] * cell_size + cell_size // 2), cell_size // 3)

    pygame.display.flip()
    pygame.time.wait(100)

if __name__ == "__main__":
    provider = IBMProvider()
    env = MarsEnv()
    agent = DQNAgent(state_size=4, action_size=4, provider=provider)

    # ✅ Embed model loading here
    import os
    if os.path.exists("best_rover_model.h5"):
        agent.model.load_weights("best_rover_model.h5")
        agent.update_target_model()
        agent.epsilon = 0.1
        print("✅ Loaded existing trained model.")
    else:
        print("🔁 No existing model found. Training from scratch.")

    for episode in range(5):  # You can increase this number
        state = env.reset()
        total_reward = 0
        for t in range(50):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            visualize(env.rover_pos, env.goal_pos, env.size, env.grid)

            if done:
                break

        agent.replay()
        agent.update_target_model()

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            agent.model.save("best_rover_model.h5")

        print(f"Episode {episode+1}: Total Reward = {round(total_reward, 2)} | Best = {round(best_total_reward, 2)}")

    pygame.quit()
