# rover-ai-qiskit


# 🚀 Mars Rover AI with Quantum-Inspired Action Selection

This project simulates a Mars rover learning to navigate a grid-like Martian terrain using **Deep Q-Learning (DQN)** and a **quantum-inspired action selection strategy** based on **Grover's algorithm** from Qiskit.

---

## 🧠 Key Features

- 🔁 **Reinforcement Learning (RL)**: Rover learns over episodes using a DQN.
- 🧮 **Quantum-Enhanced Decisions**: Action selection is guided by a simulated Grover search circuit to emphasize higher-value actions.
- 🎮 **Visualization**: Real-time display using Pygame to observe rover behavior.
- 🌌 **Goal**: Reach the target while avoiding obstacles in a grid environment.

---

## 📐 Environment Details

- Grid Size: Configurable (default = 6×6)
- Rover starts at: `(0, 0)`
- Goal position: `(size-1, size-1)`
- Obstacles: 1 random obstacle placed each episode

### Rewards:
| Condition              | Reward   |
|------------------------|----------|
| Reaching Goal          | +10      |
| Stepping into Obstacle | -5       |
| Normal Step            | -0.1     |

---

## 🧪 Quantum-Inspired Action Selection

The rover doesn't always pick the highest Q-value. Instead, it:
1. Simulates Grover’s algorithm using `qiskit`
2. Probabilistically biases toward the best actions
3. Adds an innovative twist to standard RL

> ⚠️ This uses Qiskit's **`BasicAer` simulator**, not a real quantum computer (but can be extended).

---

## 🛠 Requirements

- Python 3.10+
- TensorFlow
- Qiskit
- Pygame
- NumPy


Install dependencies:
```bash
pip install tensorflow qiskit pygame numpy
'''

# rover-ai-qiskit
# rover-ai-qiskit
# rover-ai-qiskit
# rover-ai-qiskit
# rover-ai-qiskit
