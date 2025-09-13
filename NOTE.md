# Mars Rover Simulation - Notes

## Project Overview
- Simulates a **Mars Rover** navigating an 8x8 grid with obstacles.
- Rover starts at `(0,0)` and aims to reach `(7,7)`.
- Uses **Deep Q-Learning (DQN)** for training.
- Action selection is **quantum-inspired** using Grover-like circuits (Qiskit).
- **Firebase Firestore** used to log training statistics.

---

## Technologies Used
- Python 3.11+
- Libraries:
  - `numpy`, `tensorflow`, `pygame`, `qiskit`, `firebase-admin`, `python-dotenv`
- Optional:
  - `qiskit-aer` for local quantum simulation
  - IBM Quantum Runtime for cloud quantum execution

---

## Key Concepts
1. **Environment**
   - Grid-based Mars terrain
   - Obstacles randomly placed
   - State = `[rover_x, rover_y, goal_x, goal_y]` normalized

2. **Agent**
   - Deep Q-Network
   - Epsilon-greedy exploration
   - Memory buffer with replay
   - Target network for stability

3. **Quantum-Inspired Action Selection**
   - Grover-like circuit amplifies probability of best action
   - Runs on Qiskit Aer (if installed) or classical argmax

4. **Visualization**
   - Pygame window
   - Rover = blue circle
   - Goal = green circle
   - Obstacles = black squares

5. **Training**
   - 500 episodes (default)
   - Each episode max 50 steps
   - Reward:
     - -0.1 per step
     - -5 for hitting obstacle
     - +10 for reaching goal
   - Best model saved as `best_rover_model.h5`

6. **Firebase Logging**
   - Stores episode number, epsilon, and best reward

---

## Future Ideas
- Extend to **hardware rover with sensors**
- Real **quantum-enhanced decision making**
- More complex terrain, obstacles, and multiple rovers
- Optimize Grover-inspired DQN for better training

I have surely used AI but to enhance my code and also to write this so..............bye
the primary focus is to learn
