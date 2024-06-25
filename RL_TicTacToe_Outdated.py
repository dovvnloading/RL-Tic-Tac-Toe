
import random
import numpy as np
import pygame
import threading
import queue
import time
import math
import pickle
from collections import defaultdict

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 400, 400
LINE_WIDTH = 10
SQUARE_SIZE = WIDTH // 3
CIRCLE_RADIUS = SQUARE_SIZE // 4
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4
RED = (255, 0, 0)
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

# Pygame window setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
screen.fill(BG_COLOR)

# Global variables for logging
game_logs = []
game_count = 0

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action, reward, next_state):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

class TicTacToe:
    def __init__(self):
        self.board = np.full(9, ' ')
        self.current_winner = None
        self.moves_history = []
        self.winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]

    def available_moves(self):
        return list(np.where(self.board == ' ')[0])

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            self.moves_history.append((square, letter))
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        for combo in self.winning_combinations:
            if square in combo and all(self.board[i] == letter for i in combo):
                return True
        return False

    def reset(self):
        self.board.fill(' ')
        self.current_winner = None
        self.moves_history = []

    def evaluate_state(self, player):
        opponent = 'O' if player == 'X' else 'X'
        score = 0
        
        # Check for wins
        if self.current_winner == player:
            return 1000
        elif self.current_winner == opponent:
            return -1000
        
        # Check for two-in-a-row opportunities
        for combo in self.winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == player and self.board[combo[2]] == ' ':
                score += 10
            elif self.board[combo[1]] == self.board[combo[2]] == player and self.board[combo[0]] == ' ':
                score += 10
            elif self.board[combo[0]] == self.board[combo[2]] == player and self.board[combo[1]] == ' ':
                score += 10
            
            # Check for opponent's two-in-a-row
            if self.board[combo[0]] == self.board[combo[1]] == opponent and self.board[combo[2]] == ' ':
                score -= 9
            elif self.board[combo[1]] == self.board[combo[2]] == opponent and self.board[combo[0]] == ' ':
                score -= 9
            elif self.board[combo[0]] == self.board[combo[2]] == opponent and self.board[combo[1]] == ' ':
                score -= 9
        
        # Favor center position
        if self.board[4] == player:
            score += 5
        elif self.board[4] == opponent:
            score -= 4
        
        return score

    def get_winning_move(self, player):
        for i in range(9):
            if self.board[i] == ' ':
                self.board[i] = player
                if self.winner(i, player):
                    self.board[i] = ' '
                    return i
                self.board[i] = ' '
        return None

    def get_blocking_move(self, player):
        opponent = 'O' if player == 'X' else 'X'
        return self.get_winning_move(opponent)

class QLearningAgent:
    def __init__(self, player_symbol, epsilon_start=0.99, epsilon_end=0.01, epsilon_decay=800000):
        self.q_table = defaultdict(lambda: np.zeros(9))
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.alpha_start = 0.3
        self.alpha_end = 0.01
        self.alpha_decay = 800000
        self.alpha = self.alpha_start
        self.gamma = 0.9
        self.total_reward = 0
        self.episode_rewards = []
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.episode = 0
        self.step_count = 0
        self.player_symbol = player_symbol
        self.experience_replay = ExperienceReplay(capacity=9000000)
        self.batch_size = 64
        self.action_counts = defaultdict(lambda: np.zeros(9))
        self.ucb_c = 2  # exploration parameter

    def choose_action(self, state, available_actions, game):
        state_tuple = tuple(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions), "Explorative"
        
        # Check for winning move
        winning_move = game.get_winning_move(self.player_symbol)
        if winning_move in available_actions:
            return winning_move, "Attempting to win"
        
        # Check for blocking move
        blocking_move = game.get_blocking_move(self.player_symbol)
        if blocking_move in available_actions:
            return blocking_move, "Defending"
        
        # Use UCB1 for action selection
        total_count = sum(self.action_counts[state_tuple])
        ucb_values = []
        for action in available_actions:
            q_value = self.q_table[state_tuple][action]
            count = self.action_counts[state_tuple][action]
            if count == 0:
                ucb_values.append(float('inf'))
            else:
                ucb_value = q_value + self.ucb_c * math.sqrt(math.log(total_count) / count)
                ucb_values.append(ucb_value)
        
        max_ucb = max(ucb_values)
        best_actions = [a for a, v in zip(available_actions, ucb_values) if v == max_ucb]
        chosen_action = random.choice(best_actions)
        
        self.action_counts[state_tuple][chosen_action] += 1
        return chosen_action, "Attacking"

    def update_q_table(self, state, action, reward, next_state):
        self.experience_replay.add(state, action, reward, next_state)
        
        for _ in range(min(self.batch_size, len(self.experience_replay.buffer))):
            s, a, r, ns = random.choice(self.experience_replay.buffer)
            q_next = np.max(self.q_table[tuple(ns)]) if ' ' in ns else 0
            self.q_table[tuple(s)][a] += self.alpha * (r + self.gamma * q_next - self.q_table[tuple(s)][a])
        
        self.total_reward += reward

    def reset_reward(self):
        self.total_reward = 0

    def update_episode_reward(self):
        self.episode_rewards.append(self.total_reward)
        self.reset_reward()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_start - self.episode / self.epsilon_decay)

    def update_alpha(self):
        self.alpha = max(self.alpha_end, self.alpha_start - self.episode / self.alpha_decay)

    def update_stats(self, result):
        self.episode += 1
        if result == 'Tie':
            self.ties += 1
        elif result == self.player_symbol:
            self.wins += 1
        else:
            self.losses += 1
        self.update_epsilon()
        self.update_alpha()

    def increase_step_count(self):
        self.step_count += 1

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'wins': self.wins,
                'losses': self.losses,
                'ties': self.ties,
                'episode': self.episode,
                'step_count': self.step_count,
                'action_counts': dict(self.action_counts)
            }, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(9), data['q_table'])
            self.epsilon = data['epsilon']
            self.alpha = data['alpha']
            self.wins = data['wins']
            self.losses = data['losses']
            self.ties = data['ties']
            self.episode = data['episode']
            self.step_count = data['step_count']
            self.action_counts = defaultdict(lambda: np.zeros(9), data['action_counts'])

def draw_lines():
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

def draw_figures(board):
    for square in range(9):
        row = square // 3
        col = square % 3
        if board[square] == 'X':
            pygame.draw.line(screen, CROSS_COLOR,
                             (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE),
                             (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                             CROSS_WIDTH)
            pygame.draw.line(screen, CROSS_COLOR,
                             (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                             (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE),
                             CROSS_WIDTH)
        elif board[square] == 'O':
            pygame.draw.circle(screen, CIRCLE_COLOR,
                               (int(col * SQUARE_SIZE + SQUARE_SIZE // 2), int(row * SQUARE_SIZE + SQUARE_SIZE // 2)),
                               CIRCLE_RADIUS, CIRCLE_WIDTH)

def print_stats(agent_x, agent_o):
    print(f'Agent X - Wins: {agent_x.wins}, Losses: {agent_x.losses}, Ties: {agent_x.ties}, Epsilon: {agent_x.epsilon:.3f}, Alpha: {agent_x.alpha:.3f}')
    print(f'Agent O - Wins: {agent_o.wins}, Losses: {agent_o.losses}, Ties: {agent_o.ties}, Epsilon: {agent_o.epsilon:.3f}, Alpha: {agent_o.alpha:.3f}')

def print_game_details(game_info):
    print(f'\nGame Number: {game_info["Game Number"]}')
    print(f'Winner: {game_info["Winner"]}')
    print(f'Final Board:')
    print_board(game_info['Final Board'])

def print_board(board):
    print('-------------')
    for i in range(3):
        print(f'| {board[i*3]} | {board[i*3+1]} | {board[i*3+2]} |')
        print('-------------')

class GameThread(threading.Thread):
    def __init__(self, agent_x, agent_o, game, gui_queue):
        threading.Thread.__init__(self)
        self.agent_x = agent_x
        self.agent_o = agent_o
        self.game = game
        self.gui_queue = gui_queue
        self.running = True

    def run(self):
        episodes = 1000000
        for _ in range(episodes):
            if not self.running:
                break
            play_game(self.agent_x, self.agent_o, self.game, self.gui_queue)

def get_human_move(game):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // SQUARE_SIZE
                row = y // SQUARE_SIZE
                move = row * 3 + col
                if move in game.available_moves():
                    return move
        pygame.display.update()

def play_game(agent_x, agent_o, game, gui_queue, human_player=None):
    global game_count
    game_count += 1
    game.reset()
    
    current_agent = random.choice([agent_x, agent_o])
    other_agent = agent_o if current_agent == agent_x else agent_x
    letter = 'X' if current_agent == agent_x else 'O'

    while True:
        state = tuple(game.board)
        available_actions = game.available_moves()

        if len(available_actions) == 0:
            agent_x.update_stats('Tie')
            agent_o.update_stats('Tie')
            break

        if human_player and letter == human_player:
            action = get_human_move(game)
            move_type = "Human move"
        else:
            action, move_type = current_agent.choose_action(state, available_actions, game)

        game.make_move(action, letter)
        current_agent.increase_step_count()

        gui_queue.put(('update', game.board))

        if game.current_winner:
            reward = 1
            if not human_player:
                current_agent.update_q_table(state, action, reward, game.board)
                other_agent.update_q_table(state, action, -reward, game.board)
            current_agent.update_stats(letter)
            other_agent.update_stats(letter)

            game_logs.append({
                'Game Number': game_count,
                'Winner': letter,
                'Moves': game.moves_history,
                'Step Count': current_agent.step_count,
                'Final Board': game.board.copy()
            })

            time.sleep(0.00000000001) # TIME DELAY FRAME RATE 
            print_stats(agent_x, agent_o)
            print_game_details(game_logs[-1])
            break

        else:
            if not human_player:
                reward = game.evaluate_state(letter) / 100  # Normalize the evaluation score
                current_agent.update_q_table(state, action, reward, game.board)

        # Switch players
        current_agent, other_agent = other_agent, current_agent
        letter = 'O' if letter == 'X' else 'X'

        time.sleep(0.00000000001) # TIME DELAY FRAME RATE

def play_game_evaluation(agent_x, agent_o, game, gui_queue):
    global game_count
    game_count += 1
    game.reset()
    
    current_agent = random.choice([agent_x, agent_o])
    other_agent = agent_o if current_agent == agent_x else agent_x
    letter = 'X' if current_agent == agent_x else 'O'

    print(f"\nGame {game_count} started. {'X' if current_agent == agent_x else 'O'} goes first.")

    while True:
        state = tuple(game.board)
        available_actions = game.available_moves()

        if len(available_actions) == 0:
            print("Game ended in a tie.")
            agent_x.update_stats('Tie')
            agent_o.update_stats('Tie')
            break

        action, move_type = current_agent.choose_action(state, available_actions, game)

        print(f"\n{letter}'s turn:")
        print(f"Move type: {move_type}")
        print(f"Chosen action: {action}")
        print(f"Board evaluation: {game.evaluate_state(letter)}")

        game.make_move(action, letter)
        current_agent.increase_step_count()

        gui_queue.put(('update', game.board))

        print("Current board state:")
        print_board(game.board)

        if game.current_winner:
            print(f"{letter} wins!")
            reward = 1
            current_agent.update_q_table(state, action, reward, game.board)
            other_agent.update_q_table(state, action, -reward, game.board)
            current_agent.update_stats(letter)
            other_agent.update_stats(letter)

            game_logs.append({
                'Game Number': game_count,
                'Winner': letter,
                'Moves': game.moves_history,
                'Step Count': current_agent.step_count,
                'Final Board': game.board.copy()
            })

            print_stats(agent_x, agent_o)
            break

        else:
            reward = game.evaluate_state(letter) / 100  # Normalize the evaluation score
            current_agent.update_q_table(state, action, reward, game.board)

        # Switch players
        current_agent, other_agent = other_agent, current_agent
        letter = 'O' if letter == 'X' else 'X'

        time.sleep(0.5)  # Add a small delay to make it easier to follow the game

def main():
    agent_x = QLearningAgent('X', epsilon_start=0.99, epsilon_end=0.01, epsilon_decay=800000)
    agent_o = QLearningAgent('O', epsilon_start=0.99, epsilon_end=0.01, epsilon_decay=800000)
    game = TicTacToe()
    gui_queue = queue.Queue()

    # Display menu options
    print("Welcome to Tic Tac Toe!")
    print("1. AI vs AI (Training)")
    print("2. Human vs Untrained AI")
    print("3. Load agents and continue training")
    print("4. Human vs Trained AI")
    print("5. Evaluation Mode")
    print("6. Exit")

    choice = input("Enter your choice (1-6): ")

    if choice in ['1', '3', '5']:
        if choice in ['3', '5']:
            agent_x.load('agent_x.pkl')
            agent_o.load('agent_o.pkl')
            print("Agents loaded.")

        if choice == '5':
            print("Entering Evaluation Mode...")
            episodes = int(input("Enter the number of games to evaluate: "))
            for _ in range(episodes):
                play_game_evaluation(agent_x, agent_o, game, gui_queue)
        else:
            print("Starting AI vs AI training...")
            game_thread = GameThread(agent_x, agent_o, game, gui_queue)
            game_thread.start()

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        game_thread.running = False

                try:
                    action, data = gui_queue.get_nowait()
                    if action == 'update':
                        screen.fill(BG_COLOR)
                        draw_lines()
                        draw_figures(data)
                        pygame.display.update()
                except queue.Empty:
                    pass

            game_thread.join()

    elif choice == '2' or choice == '4':
        if choice == '4':
            agent_x.load('agent_x.pkl')
            agent_o.load('agent_o.pkl')
            print("Trained agents loaded. Ready to play!")
        else:
            print("Playing against an untrained AI.")

        human_player = input("Do you want to play as X or O? ").upper()
        while human_player not in ['X', 'O']:
            human_player = input("Invalid choice. Please enter X or O: ").upper()

        ai_player = agent_o if human_player == 'X' else agent_x

        running = True
        while running:
            game.reset()
            screen.fill(BG_COLOR)
            draw_lines()
            pygame.display.update()

            current_player = 'X'
            while True:
                if current_player == human_player:
                    move = get_human_move(game)
                else:
                    state = tuple(game.board)
                    available_actions = game.available_moves()
                    move, _ = ai_player.choose_action(state, available_actions, game)

                game.make_move(move, current_player)
                screen.fill(BG_COLOR)
                draw_lines()
                draw_figures(game.board)
                pygame.display.update()

                if game.current_winner:
                    print(f"{'You' if game.current_winner == human_player else 'AI'} wins!")
                    break
                elif len(game.available_moves()) == 0:
                    print("It's a tie!")
                    break

                current_player = 'O' if current_player == 'X' else 'X'

            play_again = input("Do you want to play again? (y/n): ")
            if play_again.lower() != 'y':
                running = False

    elif choice == '6':
        print("Exiting the game.")
    else:
        print("Invalid choice. Exiting the game.")

    # Save agents before exiting
    agent_x.save('agent_x.pkl')
    agent_o.save('agent_o.pkl')
    print("Agents saved. Goodbye!")

    pygame.quit()

if __name__ == '__main__':
    main()