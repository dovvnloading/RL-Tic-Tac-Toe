This project implements an advanced Tic Tac Toe game using Pygame, featuring multiple AI agents powered by a sophisticated reinforcement learning algorithm. The core of the AI is based on a Dueling Double Q-learning approach, which is an enhancement over traditional Q-learning.
The game includes several key components:

TicTacToe class: Represents the game board and logic, including move validation, win checking, and state evaluation.
DuelingDoubleQLearningAgent class: Implements the AI player using Dueling Double Q-learning. This advanced algorithm separates state value and action advantage, potentially leading to better performance in certain scenarios. The agent uses two sets of Q-tables (value and advantage) to reduce overestimation bias.
PrioritizedExperienceReplay class: Implements experience replay with prioritized sampling, which helps the agent learn more efficiently from important experiences.
Game visualization: Utilizes Pygame to render the game board and moves, providing a graphical interface for human players.
Multiple game modes:

AI vs AI training: Allows two AI agents to play against each other to improve their strategies.
Human vs Untrained AI: Lets a human player compete against an untrained AI.
Human vs Trained AI: Enables play against a pre-trained AI agent.
Evaluation mode: Runs multiple games between AI agents to assess their performance.


Save/Load functionality: Allows saving and loading of trained agents, preserving their learned strategies.
Shaped reward system: Implements a nuanced reward structure that considers factors like board control and potential winning moves, not just game outcomes.
UCB1 exploration: Uses Upper Confidence Bound algorithm for balancing exploration and exploitation in action selection.

The project demonstrates advanced concepts in reinforcement learning, including:

Dueling network architecture
Double Q-learning
Prioritized experience replay
UCB1 exploration
Shaped rewards

It also showcases software engineering practices like modular design, multithreading for UI responsiveness, and the use of Python's queue system for thread communication.
