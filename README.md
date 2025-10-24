
<div align="center">
  
  <h1>Reinforcement Learning Tic-Tac-Toe</h1>
  <p><strong>An Interactive GUI for Training and Visualizing a Dueling Double Q-Learning Agent</strong></p>
  <p>
    <a href="https://github.com/dovvnloading/RL-Tic-Tac-Toe/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dovvnloading/RL-Tic-Tac-Toe?style=for-the-badge&color=3498db" alt="License"></a>
    <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white&color=34495e" alt="Python Version">
    <img src="https://img.shields.io/badge/Framework-PySide6-blue?style=for-the-badge&logo=qt&logoColor=white&color=27ae60" alt="Framework">
  </p>
</div>

---

## Overview

This project is more than just a game of Tic-Tac-Toe; it is a sophisticated educational tool designed to demystify the core concepts of Reinforcement Learning. It provides a tangible, interactive platform to train an advanced RL agent from scratch, observe its learning process in real-time, and directly challenge its acquired strategy.

The primary goal is to lower the barrier to entry for aspiring ML engineers and enthusiasts, transforming abstract RL theories into a concrete and engaging visual experience. By watching the agent evolve from random moves to intelligent, strategic play, users can build a more intuitive understanding of how machines learn through trial and error.

<br>

<div align="center">
  <img width="80%" alt="Dark Theme Screenshot" src="https://github.com/user-attachments/assets/06286171-34c5-4aaf-a15d-81efd07228a1" />
  <br><em> </em><br><br>
  <img width="80%" alt="Light Theme Screenshot" src="https://github.com/user-attachments/assets/3ae9f581-a0c0-4a94-8ec5-e0e519a4eca8" />
  <br><em> </em>
</div>

## Core Features

- **Interactive PySide6 GUI:** A polished, professional graphical interface with selectable light and dark themes.
- **Live AI Training:** Train two competing agents against each other for a user-specified number of games, with real-time progress updates via a progress bar and statistics panel.
- **Direct Agent Challenge:** Play directly against a trained or untrained agent to test its strategic capabilities.
- **Detailed Evaluation Mode:** Run a single, slowed-down game between two AI agents, with a move-by-move log detailing their internal evaluations and strategic choices.
- **Agent Persistence:** Save the trained state of your agents to a file and load them back in later to continue training or for immediate play.
- **Configurable Training Speed:** Utilize a speed slider to watch the training process unfold at a comprehensible pace or accelerate it for rapid learning.

## Technical Deep Dive

The intelligence of the application is powered by a sophisticated RL model, and the entire program is built upon a clean, modular architecture.

### The Reinforcement Learning Agent

The agent employs a **Dueling Double Q-Learning (DDQN)** architecture with **Prioritized Experience Replay (PER)**. This combination represents a significant advancement over a basic Q-learning model.

| Component                       | Purpose                                                                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dueling Network**             | Separates the Q-value calculation into two streams: a **Value Stream** (how good the current state is) and an **Advantage Stream** (how much better a specific action is compared to others). This leads to more robust and stable learning. |
| **Double Q-Learning**           | Mitigates the overestimation bias inherent in standard Q-learning by decoupling the action selection from the target Q-value calculation, resulting in more accurate value estimates. |
| **Prioritized Experience Replay** | Instead of sampling past experiences uniformly, PER prioritizes experiences that were surprising or led to a large error in prediction. This allows the agent to learn more efficiently from its most significant mistakes and successes. |

### Application Architecture

The project adheres to a strict **Separation of Concerns** principle, ensuring that the UI, application logic, and core AI are independently maintainable and testable.

| Module                 | Responsibility                                                                                         |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| `RL_TicTacToe.py`      | Main application entry point. Initializes and launches the PySide6 GUI.                                  |
| `gui.py`               | Contains all PySide6 widgets, window layout, and UI event handling. Manages the "View" layer.            |
| `game_controller.py`   | Orchestrates the flow of the application (training sessions, evaluation games). Acts as the "Controller".|
| `rl_agent.py`          | Defines the `DuelingDoubleQLearningAgent` and `PrioritizedExperienceReplay` classes. The "Model" (AI).    |
| `game_logic.py`        | Contains the pure `TicTacToe` game engine with board state and rules. The "Model" (Game).                |
| `themes.py`            | Defines the color palettes and generates the dynamic stylesheet for the UI.                              |

---

## Installation & Setup

To get the application running locally, follow these steps.

**1. Clone the repository:**
```bash
git clone https://github.com/dovvnloading/RL-Tic-Tac-Toe.git
cd RL-Tic-Tac-Toe
```

**2. Create and activate a virtual environment:**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the required dependencies:**
A `requirements.txt` file is provided for convenience.
```bash
pip install -r requirements.txt
```

**4. Launch the application:**
```bash
python RL_TicTacToe.py
```

## How to Use the Application

The interface is designed to be intuitive and is segmented into three main modes of operation, accessible from the "Controls" tab.

1.  **Training the Agent:**
    -   Select the number of games for the training session using the "Games" spinbox.
    -   Adjust the "Speed" slider to control the delay between games (left is fastest, right is slowest).
    -   Click **"Start Training"**. The progress bar will activate, and the Log and Statistics tabs will update periodically. You can stop the training at any time.

2.  **Playing Against the Agent:**
    -   First, ensure you have either trained a new agent or loaded a pre-trained one using the "Load" button.
    -   Click **"Play as X"** or **"Play as O"** to begin a game. The board will become active when it's your turn.

3.  **Evaluating Agent Performance:**
    -   Click **"Run Evaluation Game"** to pit two agents against each other.
    -   The game will proceed automatically at a slow pace.
    -   Navigate to the **"Log"** tab to see a detailed, turn-by-turn breakdown of each agent's decision-making process.

## Project Contents

This repository includes everything needed to run, inspect, and experiment with the project.

-   **/ (root):** Contains the core Python source files (`.py`).
-   **`.assets/`:** Contains media used for the README file.
-   **`trained_agents/`:** Includes pre-trained `agent_x.pkl` and `agent_o.pkl` files after ~100,000 games, ready for immediate loading and play.
-   **`RL-Tic-Tac-Toe.sln`:** A Visual Studio solution file for easy project management in the VS IDE.
-   **`LICENSE`:** The MIT License file.
-   **`requirements.txt`:** A list of all Python dependencies.

---

<div align="center">
  This project was developed as a tool to bridge the gap between theoretical knowledge and practical application in Reinforcement Learning.
</div>
```
