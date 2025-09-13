import numpy as np
import tensorflow as tf
from qiskit import QuantumCircuit
from qiskit import transpile

# try to import IBM runtime optional but we will not use in local simulation
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    HAVE_IBM_RUNTIME = True
except Exception:
    HAVE_IBM_RUNTIME = False

# Aer simulator local quantum runs ke liye
try:
    from qiskit_aer import AerSimulator
    HAVE_AER = True
except Exception:
    HAVE_AER = False

import random
import pygame
import sys
from collections import deque
import math
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# .env load karo aur token uthao
load_dotenv()
token = os.getenv("IBM_QUANTUM_TOKEN")

if HAVE_IBM_RUNTIME and token:
    try:
        QiskitRuntimeService.save_account(channel="ibm_quantum", token=token, overwrite=True)
    except Exception:
        # if save fails then ..........
        pass

# Firebase setup  firebase-key.json is must to be there in the folder
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

best_total_reward = float('-inf') # foir saving best model

# optional IBM service maybe will work ith better hardware
#TODO:give it a check try google collab for cloud GPU services

 
def get_ibm_service():
    if not HAVE_IBM_RUNTIME:
        return None
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
    except Exception:
        print("[ERROR] IBM Runtime service nahi mila Token set karo ya .env me daalo")
        token_input = input("IBM Quantum API Token ya enter skip ke liye").strip()
        if token_input:
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=token_input, overwrite=True)
            service = QiskitRuntimeService()
        else:
            return None
    return service

# Mars environment define using pygaem
class MarsEnv:
    def __init__(self, size=8, num_obstacles=10):
        self.size = size
        self.num_obstacles = num_obstacles
        self.reset() # initialize karo

    # environment reset
    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.rover_pos = [0, 0] 
        self.goal_pos = [self.size - 1, self.size - 1] 
        placed = 0
        while placed < self.num_obstacles:
            x, y = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
            if [x, y] != self.rover_pos and [x, y] != self.goal_pos and self.grid[x, y] != -1:
                self.grid[x, y] = -1 # obstacle place karo
                placed += 1
        return self.get_state()

    # state normalize then send to network
    def get_state(self):
        return np.array(self.rover_pos + self.goal_pos, dtype=np.float32) / (self.size - 1)

    # reward and moves 
    def step(self, action):
        x, y = self.rover_pos
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
        dx, dy = moves[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            if self.grid[new_x, new_y] == -1:
                return self.get_state(), -5, True # obstacle hit then end of one episode

            self.rover_pos = [new_x, new_y]

        done = self.rover_pos == self.goal_pos
        reward = 10 if done else -0.1 # goal reach +10 har step -0.1
        return self.get_state(), reward, done

# quantum inspired action selection using grover like circuit
def grover_select_action(q_values):
    original_len = len(q_values)
    if original_len == 0:
        return 0
    # pad to power of 2 for quantum circuit
    padded_len = 2 ** math.ceil(math.log2(original_len))
    padded_q = np.zeros(padded_len)
    padded_q[:original_len] = q_values
    best_action = int(np.argmax(padded_q))
    n = int(math.log2(padded_len))

    # circuit build with superposition oracle diffusion#TODO: GIVE IT A DETAILED STUDY MORE THAN WHAT IS KNOWN
    qc = QuantumCircuit(n, n)
    qc.h(list(range(n))) # superposition

    bin_str = format(best_action, f'0{n}b')
    # oracle mark best_action
    for i, bit in enumerate(bin_str):
        if bit == '0':
            qc.x(i)
    if n > 1:
        qc.h(n -1)
        qc.mcx(list(range(n -1)),n-1)
        qc.h(n -1)
    else:
        qc.z(0)
    for i, bit in enumerate(bin_str):
        if bit =='0':
            qc.x(i)

    # diffusion amplifys probability
    qc.h(list(range(n)))
    qc.x(list(range(n)))
    if n > 1:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    else:
        qc.z(0)
    qc.x(list(range(n)))
    qc.h(list(range(n)))

    qc.measure(list(range(n)), list(range(n)))

    # Aer me run if available else classical argmax#TODO: needs improvement
    if HAVE_AER:
        sim = AerSimulator()
        tqc = transpile(qc, sim)
        job = sim.run(tqc, shots=128)
        try:
            result = job.result()
            counts = result.get_counts()
            measured = int(max(counts, key=counts.get),2)
            return measured if measured < original_len else best_action
        except Exception as e:
            print("[WARN] Aer execution failed", e)
            return best_action
    else:
        print("[WARN] qiskit-aer nahi hai using classical argmax")
        return best_action


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) #
        self.gamma = 0.99 
        self.epsilon = 1.0 # exploration
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.model = self.build_model() #main_build
        self.target_model = self.build_model()
        self.update_target_model()

    # keras model build
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # epsilon greedy action chooses
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1) # explore
        q_vals = self.model.predict(np.array([state]), verbose=0)[0]
        return grover_select_action(q_vals) # quantum inspired

    # memory me store experience
    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    # minibatch se train  model
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


def visualize(rover_pos, goal_pos, grid_size, grid, screen, cell_size):
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

    pygame.draw.circle(screen, (0, 0, 255), (env.rover_pos[1] * cell_size + cell_size // 2, env.rover_pos[0] * cell_size + cell_size // 2), cell_size // 3)
    pygame.draw.circle(screen, (0, 255, 0), (env.goal_pos[1] * cell_size + cell_size // 2, env.goal_pos[0] * cell_size + cell_size // 2), cell_size // 3)

    pygame.display.flip()
    pygame.time.wait(100)


if __name__ == "__main__":
    ibm_service = get_ibm_service = None
    if HAVE_IBM_RUNTIME:
        try:
            ibm_service = QiskitRuntimeService()
        except Exception:
            ibm_service = None

    env = MarsEnv(size=8, num_obstacles=10)
    agent = DQNAgent(state_size=4, action_size=4)

    pygame.init()
    cell_size = 500 // env.size
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Mars Rover Simulation")

    # load model agar hai
    if os.path.exists("best_rover_model.h5"):
        agent.model.load_weights("best_rover_model.h5")
        agent.update_target_model()
        agent.epsilon = 0.1
        print(" Loaded existing trained model")
    else:
        print("No existing model found Training from scratch")

   
    for episode in range(500):
        state = env.reset()
        total_reward = 0
        for t in range(50):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            visualize(env.rover_pos, env.goal_pos, env.size, env.grid, screen, cell_size)

            if done:
                break

        agent.replay()
        agent.update_target_model()

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            agent.model.save("best_rover_model.h5") # best model save

        print(f"Episode {episode+1} Total Reward = {round(total_reward, 2)} | Best = {round(best_total_reward, 2)}")

        #using firebase to save model
        db.collection("rover_training").document(f"episode_{episode+1}").set({
            "episode": episode + 1,
            "epsilon": float(agent.epsilon),
            "best_reward": float(best_total_reward)
        })

    pygame.quit()
s
