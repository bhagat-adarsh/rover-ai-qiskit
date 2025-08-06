import numpy as np
import tensorflow as tf
from qiskit import QuantumCircuit
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_provider.accounts.exceptions import AccountNotFoundError
import random
import pygame
import sys
from collections import deque
import math
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Load .env and get IBM token
load_dotenv()
token = os.getenv("IBM_QUANTUM_TOKEN")

# Initialize Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Store best reward
###global best_total_reward
best_total_reward = float('-inf')

# extracting token from THe IBM Quantum account
def get_ibm_provider():
    try:
        provider = IBMProvider()
        # Check if the account is saved
    except AccountNotFoundError:
        #TODO: remove the input prompt in production
        print("[ERROR] IBM Quantum account not found.")
        print("Please paste your IBM Quantum API token below to save it:")
        token = input("IBM Quantum API Token: ").strip()
        IBMProvider.save_account(token, overwrite=True)
        print("\u2705 IBM Quantum account saved. Re-running with credentials...\n")
        provider = IBMProvider()
    return provider


#DEFines the envoirment for the stimulation jaise start position goal kya hai and all+obstacles
class MarsEnv:
    def __init__(self, size=8, num_obstacles=10):
        self.size = size
        self.num_obstacles = num_obstacles
        self.reset()
    #Intitialising the window
    #pygame ki tarah
    #start top left corner GOAL FIXED AT BOTTOM RANDOM OBSTACLE HOGA GENERATE BICH MAI
    #TODO : Obstacles needs to be increased 
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
    #action kai bad step analyze karna hai
    def step(self, action):
        x, y = self.rover_pos
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = moves[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            if self.grid[new_x, new_y] == -1:
                return self.get_state(), -5, True

            self.rover_pos = [new_x, new_y]

        done = self.rover_pos == self.goal_pos
        reward = 10 if done else -0.1
        return self.get_state(), reward, done
    
    #+10 reward if the rover reaches the goal

    #-0.1 penalty for each step (to encourage faster solutions)

    #-5 penalty if the rover hits an obstacle (and episode ends)


#usecase of the grovers algorithm to slect tjhe best action based on Q-values
#grover algorithm here  finda a target item faster then a classical one in a unsorted list of size 'N' O( root N​ ) time
#sada states ko equally liya (superposition)
#Superposition: Prepare all possible states equally.

#Oracle: Flip the sign of the target state (marking it). Amplification (Diffusion): Amplify the marked state’s probability.Repeat steps 2 and 3 about 𝑁 N  times, then measure
def grover_select_action(q_values, provider):
    original_len = len(q_values)
    padded_len = 2 **math.ceil(math.log2(original_len))#Quantum circuits need sizes that are powers of 2.
    padded_q = np.zeros(padded_len)
    padded_q[:original_len] =q_values
    best_action = int(np.argmax(padded_q))
    n = int(math.log2(padded_len))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    bin_str = format(best_action, f'0{n}b')#ORAcle: Convert the best action to binary string with padding.
    for i, bit in enumerate(bin_str):
        if bit == '0':
            qc.x(i)
    if n> 1:
        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)
        qc.h(n-1)
    else:
        qc.z(0)
    for i, bit in enumerate(bin_str):
        if bit =='0':
            qc.x(i)

    qc.h(range(n))
    qc.x(range(n))
    if n > 1:
        qc.h(n - 1)
        qc.mcx(list(range(n-1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))

    qc.measure(range(n), range(n))

    try:
        backend = provider.get_backend("ibm_mumbai")#TODO: NEEDS TO BE CHANGED UNABLE TO ACCESS IBM BACKEND
    except Exception as e:
        print("[ERROR] Could not access IBM backend:", e)
        return best_action
    job = backend.run(qc, shots=1)
    result = job.result()
    counts = result.get_counts()
    action = int(max(counts, key=counts.get), 2)

    return action if action < original_len else best_action
#A neural network to estimate the Q-values
class DQNAgent:
    def __init__(self, state_size, action_size, provider):
        self.state_size = state_size
        self.action_size = action_size
        self.provider = provider
        self.memory = deque(maxlen=2000)#past ko yaad rakh kai galti na kare
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        #exploration or exploitation dono ko control karega
        self.model = self.build_model()#main Q-network
        self.target_model = self.build_model()#slower one to stabalize training
        self.update_target_model()
#building the model using keras
#model sai training model mai copy 
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

    def act(self, state):#Chooses an action using epsilon greedy policyl; RL
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)#random action for exploration
        q_vals = self.model.predict(np.array([state]), verbose=0)[0]
        return grover_select_action(q_vals, self.provider)#else the grover algorithm to select the best action based on Q-values

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))#stores the experience in memory

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
#viSUALISATION USIG PYGAME
#TODO:make a 3d stimulation and a use matplotlib to trrack its progress and reprentst using a graph
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

    pygame.draw.circle(screen, (0, 0, 255), (rover_pos[1] * cell_size + cell_size // 2, rover_pos[0] * cell_size + cell_size // 2), cell_size // 3)
    pygame.draw.circle(screen, (0, 255, 0), (goal_pos[1] * cell_size + cell_size // 2, goal_pos[0] * cell_size + cell_size // 2), cell_size // 3)

    pygame.display.flip()
    pygame.time.wait(100)

if __name__ == "__main__":
    provider = get_ibm_provider()
    env = MarsEnv(size=8, num_obstacles=10)
    agent = DQNAgent(state_size=4, action_size=4, provider=provider)

    pygame.init()
    cell_size = 500 // env.size
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Mars Rover Simulation")

    if os.path.exists("best_rover_model.h5"):
        agent.model.load_weights("best_rover_model.h5")
        agent.update_target_model()
        agent.epsilon = 0.1
        print("\u2705 Loaded existing trained model.")
    else:
        print("\ud83d\udd01 No existing model found. Training from scratch.")

    for episode in range(5):
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
            agent.model.save("best_rover_model.h5")

        print(f"Episode {episode+1}: Total Reward = {round(total_reward, 2)} | Best = {round(best_total_reward, 2)}")

        db.collection("rover_training").document("progress").set({
            "episode": episode + 1,
            "epsilon": float(agent.epsilon),
            "best_reward": float(best_total_reward)
        })

    pygame.quit()
#TODO: multiple files mai later dhift kardena functions and sdifreent classes ko
