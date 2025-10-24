import numpy as np

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
        
        if self.current_winner == player:
            return 1000
        elif self.current_winner == opponent:
            return -1000
        
        for combo in self.winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == player and self.board[combo[2]] == ' ':
                score += 10
            elif self.board[combo[1]] == self.board[combo[2]] == player and self.board[combo[0]] == ' ':
                score += 10
            elif self.board[combo[0]] == self.board[combo[2]] == player and self.board[combo[1]] == ' ':
                score += 10
            
            if self.board[combo[0]] == self.board[combo[1]] == opponent and self.board[combo[2]] == ' ':
                score -= 9
            elif self.board[combo[1]] == self.board[combo[2]] == opponent and self.board[combo[0]] == ' ':
                score -= 9
            elif self.board[combo[0]] == self.board[combo[2]] == opponent and self.board[combo[1]] == ' ':
                score -= 9
        
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