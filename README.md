# Reinforcement Learning Tic-Tac-Toe

This project implements a Tic-Tac-Toe game where AI agents learn to play using advanced Reinforcement Learning techniques. The application features a graphical user interface built with Pygame and allows for agent training, evaluation, and gameplay against a human opponent. The agents are trained using a Dueling Double Q-Learning algorithm combined with Prioritized Experience Replay for efficient learning.

## Features

-   **AI vs. AI Training Mode**: Run simulations of two AI agents playing against each other for thousands of episodes to train them from scratch.
-   **Human vs. AI Gameplay**: Play against a pre-trained or an untrained AI agent.
-   **Agent Persistence**: The trained state of the agents (their learned Q-tables) can be saved to and loaded from `.pkl` files, allowing for training to be paused and resumed.
-   **Evaluation Mode**: Observe trained agents play against each other with detailed console output for each move, providing insight into their decision-making process.
-   **Graphical User Interface**: A clean and responsive GUI built with Pygame to visualize the game in real-time.
-   **Multi-threaded Training**: The AI training process runs on a separate thread, keeping the GUI responsive.

## Core Reinforcement Learning Concepts

This project utilizes several advanced reinforcement learning concepts to train the agents effectively:

-   **Dueling Double Q-Learning (DDQN)**: The core learning algorithm.
    -   **Double Q-Learning**: It uses two separate Q-networks (in this case, Q-tables) to decouple action selection from target Q-value generation. This helps to reduce the overestimation bias common in standard Q-learning.
    -   **Dueling Network Architecture**: The Q-value for a state-action pair is decomposed into two parts: the state-value function `V(s)` and the action-advantage function `A(s, a)`. This allows the agent to learn which states are valuable without having to learn the effect of each action at each state.

-   **Prioritized Experience Replay (PER)**: Instead of sampling experiences uniformly from a replay buffer, PER samples experiences based on their TD-error. Experiences that are more "surprising" (i.e., have a higher TD-error) are replayed more frequently, leading to more efficient learning.

-   **Reward Shaping**: The agents receive more nuanced rewards than just win/loss/draw. Intermediate rewards are given for strategically valuable moves, such as taking the center square, creating a two-in-a-row threat, or blocking an opponent's threat. This guides the agents toward learning effective strategies more quickly.

-   **UCB1 Exploration**: For action selection during exploitation, the Upper Confidence Bound (UCB1) algorithm is used. This provides a sophisticated balance between exploring less-visited actions and exploiting actions that are known to be good.

## Requirements

-   Python 3.x
-   Pygame
-   NumPy

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/dovvnloading/RL-Tic-Tac-Toe.git
    ```

2.  Navigate to the project directory:
    ```bash
    cd RL-Tic-Tac-Toe
    ```

3.  Install the required Python packages:
    ```bash
    pip install pygame numpy
    ```

## Usage

Run the main script from your terminal:

```bash
python RL_TicTacToe.py
```

You will be presented with a menu of options in the terminal:

1.  **AI vs AI (Training)**: Starts a training session where two AI agents play against each other. The Pygame window will show the games being played at high speed. The console will print statistics after each game. Close the Pygame window to stop training. The agents' progress will be saved automatically to `agent_x.pkl` and `agent_o.pkl`.

2.  **Human vs Untrained AI**: Play a game against an AI agent that has not undergone any training.

3.  **Load agents and continue training**: Loads the previously saved progress from `agent_x.pkl` and `agent_o.pkl` and continues the AI vs AI training session.

4.  **Human vs Trained AI**: Loads trained agents from the `.pkl` files and allows you to play against one of them. For the best experience, run the training mode (Option 1 or 3) first.

5.  **Evaluation Mode**: Loads trained agents and runs a specified number of games between them. This mode prints detailed information to the console about each agent's move, including the reason for the move (e.g., "Attempting to win", "Defending", "Attacking") and the board evaluation score. Games run at a slower pace to allow for observation.

6.  **Exit**: Closes the application.

### Gameplay Controls

When playing against the AI, click on any empty square on the board to make your move.

## File Structure

-   `RL_TicTacToe.py`: The single Python script containing all the game logic, agent implementation, and GUI code.
-   `agent_x.pkl` (generated): The saved state and learned Q-tables for the agent playing as 'X'.
-   `agent_o.pkl` (generated): The saved state and learned Q-tables for the agent playing as 'O'.
