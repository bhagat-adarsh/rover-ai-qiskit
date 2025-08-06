**Quantum Snake AI with IBM Quantum Integration: Environment Setup & Explanation**

---

### 🚀 Objective

To create a reinforcement learning-based Snake game that integrates quantum-enhanced action selection using **Grover's algorithm**, executed on IBM's free quantum computers (e.g., `ibm_mumbai`).

---

### 🌐 Key Technologies Used

| Technology          | Role                                                              |
| ------------------- | ----------------------------------------------------------------- |
| Python 3.10         | Programming language                                              |
| Qiskit              | Quantum programming toolkit from IBM                              |
| qiskit-aer          | Simulated quantum backend                                         |
| qiskit-ibm-provider | IBM Quantum backend provider (for access to real quantum devices) |
| TensorFlow          | For implementing reinforcement learning (Q-learning)              |
| Pygame              | For rendering the Snake game                                      |
| python-dotenv       | Load IBM Quantum API token from a secure .env file                |

---

### 🌍 Environment Setup (Detailed)

#### 1. **Create a Clean Python Environment**

```bash
python3.10 -m venv qvenv
source qvenv/bin/activate
```

This prevents conflicts between different versions of packages like Qiskit and TensorFlow.

#### 2. **Install Compatible Packages**

```bash
pip install --upgrade pip

pip install \
  qiskit==2.0.2 \
  qiskit-aer==0.13.3 \
  qiskit-ibm-provider==0.11.0 \
  pygame==2.5.2 \
  tensorflow==2.19.0 \
  python-dotenv==1.0.1
```

Each version is chosen for compatibility with Python 3.10 and Qiskit >= 1.0 standards.

#### 3. **.env File for IBM Quantum Token**

Create a file named `.env` in your project root:

```
IBM_QUANTUM_TOKEN="your_ibm_quantum_api_token_here"
```

* Enclose the token in double quotes if it includes special characters.
* This keeps your credentials secure and out of source code.

---

### 🌐 Qiskit Quantum Access

#### IBM Quantum Account Setup:

```python
from qiskit_ibm_provider import IBMProvider
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("IBM_QUANTUM_TOKEN")
provider = IBMProvider(token=api_token)
backend = provider.get_backend("ibm_mumbai")
```

* Loads your token securely
* Instantiates the provider
* Selects a real quantum backend (`ibm_mumbai`)

---

### 🏋️️ Training Optimization (Avoiding Retraining)

To persist model learning across sessions:

#### Saving:

```python
model.save_weights("model_weights.h5")
```

#### Loading:

```python
if os.path.exists("model_weights.h5"):
    model.load_weights("model_weights.h5")
```

* Ensures your AI doesn't retrain every time the game restarts

---

### ⚖️ Summary

This environment enables you to:

* Train an RL-based Snake agent
* Offload decision-making to Grover's quantum search
* Execute on real IBM Quantum hardware for free

Perfect for hybrid quantum-classical reinforcement learning research and demos.
