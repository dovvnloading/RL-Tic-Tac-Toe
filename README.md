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




Step-by-Step Guide: Using the Tic Tac Toe AI Project
Prerequisites

Python installed on your system
Pygame library installed (pip install pygame)
The project code loaded into your Python IDE

Steps to Run and Use the Program

Run the Script

Execute the main script in your IDE.
The program will start and display a menu in the console.


Choose a Game Mode
When prompted, enter a number (1-6) to select a game mode:

AI vs AI (Training)
Human vs Untrained AI
Load agents and continue training
Human vs Trained AI
Evaluation Mode
Exit


For AI vs AI Training (Option 1)

The training will start automatically.
You'll see the game board updating in a Pygame window.
Training statistics will be printed in the console.
Close the Pygame window to stop training and save the agents.


For Human vs Untrained AI (Option 2)

Choose to play as X or O when prompted.
The game board will appear in a Pygame window.
Click on the board to make your move when it's your turn.
The AI will automatically make its move.
After the game ends, choose to play again or exit.


For Loading Agents and Continuing Training (Option 3)

The program will load previously saved agents.
Training will continue as in Option 1.


For Human vs Trained AI (Option 4)

The program will load previously trained agents.
Follow the same steps as in Option 2 to play against the trained AI.


For Evaluation Mode (Option 5)

Enter the number of games you want the AI to play for evaluation.
The program will run these games and display detailed statistics.


Exiting the Program (Option 6)

Choose this option to exit the program.
The AI agents will be automatically saved before exiting.



Additional Notes

During AI training or evaluation, you can close the Pygame window at any time to stop the process and save the agents.
After playing games or training, the AI agents are automatically saved for future use.
You can always load previously trained agents using Option 3 or 4.
