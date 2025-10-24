import random
import time

class GameController:
    def __init__(self, agent_x, agent_o, game_logic):
        self.agent_x = agent_x
        self.agent_o = agent_o
        self.game = game_logic
        self._is_running = True

    def stop(self):
        self._is_running = False

    def shaped_reward(self, game, player):
        base_reward = game.evaluate_state(player) / 100
        if game.board[4] == player: base_reward += 0.1
        for combo in game.winning_combinations:
            if sum(game.board[i] == player for i in combo) == 2 and sum(game.board[i] == ' ' for i in combo) == 1:
                base_reward += 0.2
        opponent = 'O' if player == 'X' else 'X'
        for combo in game.winning_combinations:
            if sum(game.board[i] == opponent for i in combo) == 2 and sum(game.board[i] == ' ' for i in combo) == 1:
                base_reward -= 0.3
        return base_reward

    def run_training_session(self, episodes, callbacks):
        for episode in range(1, episodes + 1):
            if not self._is_running:
                break

            self.game.reset()
            current_agent, other_agent = (self.agent_x, self.agent_o) if random.choice([True, False]) else (self.agent_o, self.agent_x)
            letter = current_agent.player_symbol

            while self._is_running:
                state = tuple(self.game.board)
                available_actions = self.game.available_moves()

                if not available_actions:
                    self.agent_x.update_stats('Tie')
                    self.agent_o.update_stats('Tie')
                    break

                action, _ = current_agent.choose_action(state, available_actions, self.game)
                self.game.make_move(action, letter)
                current_agent.increase_step_count()

                if self.game.current_winner:
                    reward = 1
                    current_agent.update_q_table(state, action, reward, self.game.board)
                    other_agent.update_q_table(state, action, -reward, self.game.board)
                    current_agent.update_stats(letter)
                    other_agent.update_stats(letter)
                    break
                else:
                    reward = self.shaped_reward(self.game, letter)
                    current_agent.update_q_table(state, action, reward, self.game.board)

                current_agent, other_agent = other_agent, current_agent
                letter = current_agent.player_symbol

            callbacks['on_progress'](episode, episodes)
            if episode % 50 == 0:
                callbacks['on_board_update'](list(self.game.board))
                callbacks['on_stats_update'](self.agent_x, self.agent_o)

            if callbacks['get_delay']() > 0:
                time.sleep(callbacks['get_delay']())
        
        message = f"Training complete ({episodes} games)." if self._is_running else "Training stopped."
        callbacks['on_finish'](message)

    def run_evaluation_game(self, callbacks):
        self.game.reset()
        current_agent, other_agent = (self.agent_x, self.agent_o) if random.choice([True, False]) else (self.agent_o, self.agent_x)
        letter = current_agent.player_symbol
        callbacks['on_log'](f"--- Starting Evaluation Game ---\nAgent {letter} starts.")
        callbacks['on_board_update'](list(self.game.board))
        time.sleep(1)

        while self._is_running:
            state = tuple(self.game.board)
            available_actions = self.game.available_moves()
            if not available_actions:
                callbacks['on_log']("Result: Game ended in a Tie.")
                break
            
            action, move_type = current_agent.choose_action(state, available_actions, self.game)
            log_msg = (f"\n{letter}'s turn:\n"
                       f"  - Move Type: {move_type}\n"
                       f"  - Chosen Action: {action}\n"
                       f"  - Board Evaluation: {self.game.evaluate_state(letter)}")
            callbacks['on_log'](log_msg)
            self.game.make_move(action, letter)
            callbacks['on_board_update'](list(self.game.board))
            
            winning_combo = self._find_winning_combo()
            if winning_combo:
                callbacks['on_win'](winning_combo)
                callbacks['on_log'](f"\nResult: {letter} wins!")
                time.sleep(1)
                break
            
            current_agent, other_agent = other_agent, current_agent
            letter = current_agent.player_symbol
            time.sleep(0.75)

        callbacks['on_finish']("Evaluation game finished.")
        callbacks['on_log'](f"--- End of Evaluation Game ---")

    def _find_winning_combo(self):
        for combo in self.game.winning_combinations:
            if self.game.board[combo[0]] == self.game.board[combo[1]] == self.game.board[combo[2]] != ' ':
                return combo
        return None