# Mars Rover Simulation with Quantum-Inspired Reinforcement Learning  

This project is a **Mars Rover simulation environment** built using Python, Pygame, TensorFlow, and Qiskit. It implements a **Deep Q-Network (DQN)** agent combined with a **Grover-inspired quantum search algorithm** for action selection.  

Although currently a **software simulation**, this is an early step toward creating a hardware rover with real-world sensors and quantum-inspired decision-making capabilities.  

---

### Features  
- **Custom Mars environment** with obstacles, rover, and goal position.  
- **Deep Q-Network (DQN)** training using TensorFlow/Keras.  
- **Quantum-inspired action selection** with Grover-like amplitude amplification (via Qiskit).  
- **Exploration vs. exploitation** handled by epsilon-greedy + quantum maximum search.  
- **Visualization** using Pygame for real-time rover movement.  
- **Model persistence** — best trained model is saved (`best_rover_model.h5`).  
- **Firebase integration** for logging episode results (epsilon, rewards, progress).  
- **Optional IBM Quantum backend** (through QiskitRuntimeService) if IBM token is provided.  

---

### Requirements 
check the requirement.txt to know all the requirements or in the project folder where one has cloned the repository will simply setup a venv (optional but recommanded if you are on any linux or mac distro )
and run pip install -r requirements.txt

Main dependencies include:  
- `numpy`  
- `tensorflow`  
- `qiskit` and optionally `qiskit-aer`  
- `pygame`  
- `firebase-admin`  
- `python-dotenv`  

#### 3. Firebase setup  
- Place your Firebase service account key in the project root as `firebase-key.json`.  
- Ensure Firestore is enabled in your Firebase project.  

#### 4. IBM Qiskit token (optional)  
- Create a `.env` file with your IBM Quantum API token:  

- The rover starts at (0,0) which isd the usual top-left for pygame you would surely know, and it aims for the bottom-right goal. 
- Obstacles are randomly generated.  
- The agent trains over multiple episodes and saves the best model automatically.  
- Real-time rover movements are displayed via Pygame.  

---

### Project Structure  
rover-quantum-sim/
│── main.py # Main simulation code
│── firebase-key.json # Firebase credentials (not included in repo)
│── .env # IBM Quantum API Token
│── best_rover_model.h5 # Saved trained model (after training)
│── requirements.txt
│── README.md
  


HMmm maybe in future i will try to make one with hardware and use a rasberipy for it and will try to learn more thankyou...., Dhanyawaad, chalo kalti ...............bye.
signing off 
         -- Alive -human
