import qtawesome as qta
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QGridLayout, QFrame, QFileDialog, QMessageBox, QGroupBox, QTextEdit, 
                             QSlider, QTabWidget, QStatusBar, QStackedLayout, QSpinBox, 
                             QProgressBar, QGraphicsDropShadowEffect, QComboBox)
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtCore import (Qt, QThread, QObject, Signal, Slot, QPropertyAnimation, QEasingCurve, 
                          QPoint, Property)

from game_logic import TicTacToe
from rl_agent import DuelingDoubleQLearningAgent
from game_controller import GameController
from themes import get_stylesheet

class CellWidget(QPushButton):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.mark = ' '
        self.setFixedSize(120, 120)

    def set_mark(self, mark):
        if self.mark == mark:
            return
        self.mark = mark
        self.setProperty("mark", mark)
        self.setText(mark if mark != ' ' else '')
        self.style().unpolish(self)
        self.style().polish(self)

        if mark != ' ':
            self.animate_mark()

    def animate_mark(self):
        anim = QPropertyAnimation(self, b"font_size")
        anim.setDuration(250)
        anim.setStartValue(10)
        anim.setEndValue(70)
        anim.setEasingCurve(QEasingCurve.OutBounce)
        anim.start()

    def get_font_size(self):
        return self.font().pointSize()

    def set_font_size(self, size):
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)

    font_size = Property(int, get_font_size, set_font_size)

class WinningLineOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.opacity = 0.0
        self.animation = QPropertyAnimation(self, b"line_opacity")
        self.animation.setDuration(400)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

    def paintEvent(self, event):
        if self.opacity > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor("#f1c40f"), 15, Qt.SolidLine, Qt.RoundCap)
            pen.setColor(QColor(241, 196, 15, int(self.opacity * 255)))
            painter.setPen(pen)
            painter.drawLine(self.start_point, self.end_point)

    def draw_line(self, start_pos, end_pos):
        self.start_point = start_pos
        self.end_point = end_pos
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.start()

    def clear(self):
        self.animation.setStartValue(self.opacity)
        self.animation.setEndValue(0.0)
        self.animation.start()

    def get_line_opacity(self):
        return self.opacity
    
    def set_line_opacity(self, opacity):
        self.opacity = opacity
        self.update()

    line_opacity = Property(float, get_line_opacity, set_line_opacity)


class GameBoardWidget(QFrame):
    cellClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cells = []
        self.winning_overlay = WinningLineOverlay(self)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        for i in range(9):
            cell = CellWidget(i)
            cell.clicked.connect(lambda checked=False, index=i: self.cellClicked.emit(index))
            self.cells.append(cell)
            grid_layout.addWidget(cell, i // 3, i % 3)

        main_layout = QStackedLayout(self)
        container = QWidget()
        container.setLayout(grid_layout)
        main_layout.addWidget(container)
        main_layout.addWidget(self.winning_overlay)

    def set_board_state(self, board_state):
        for i, mark in enumerate(board_state):
            self.cells[i].set_mark(mark)

    def set_enabled(self, enabled):
        for cell in self.cells:
            if cell.mark == ' ':
                cell.setEnabled(enabled)

    def show_winning_line(self, combination):
        cell_size = self.cells[0].width()
        start_cell_pos = self.cells[combination[0]].pos()
        end_cell_pos = self.cells[combination[2]].pos()
        center_offset = QPoint(cell_size // 2, cell_size // 2)
        start_pos = start_cell_pos + center_offset
        end_pos = end_cell_pos + center_offset
        self.winning_overlay.draw_line(start_pos, end_pos)

    def reset(self):
        self.winning_overlay.clear()
        for cell in self.cells:
            cell.set_mark(' ')


class GameWorker(QObject):
    boardUpdated = Signal(list)
    statsUpdated = Signal(object, object)
    finished = Signal(str)
    logMessage = Signal(str)
    stateChanged = Signal(str)
    gameWon = Signal(list)
    trainingProgress = Signal(int, int)

    def __init__(self, agent_x, agent_o):
        super().__init__()
        self.game_logic = TicTacToe()
        self.controller = GameController(agent_x, agent_o, self.game_logic)
        self.mode = "train"
        self.human_player = None
        self.training_episodes = 0
        self._delay = 0.0

    def stop(self):
        self.controller.stop()

    def set_mode(self, mode, human_player=None, episodes=0):
        self.mode = mode
        self.human_player = human_player
        self.training_episodes = episodes

    @Slot(int)
    def set_delay(self, slider_value):
        self._delay = (slider_value / 100.0) ** 2

    def get_delay(self):
        return self._delay

    @Slot(int)
    def human_made_move(self, move):
        if self.current_player == self.human_player and self.game_logic.board[move] == ' ':
            self.game_logic.make_move(move, self.human_player)
            self.boardUpdated.emit(list(self.game_logic.board))
            if self._check_game_over(): return
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            self.stateChanged.emit("PLAYING_AI")
            QThread.msleep(500)
            self._ai_move()
            
    def _ai_move(self):
        ai_agent = self.controller.agent_x if self.current_player == 'X' else self.controller.agent_o
        state = tuple(self.game_logic.board)
        available_actions = self.game_logic.available_moves()
        action, _ = ai_agent.choose_action(state, available_actions, self.game_logic)
        self.game_logic.make_move(action, self.current_player)
        self.boardUpdated.emit(list(self.game_logic.board))
        if self._check_game_over(): return
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        self.stateChanged.emit("PLAYING_HUMAN")
    
    def _check_game_over(self):
        winning_combo = self.controller._find_winning_combo()
        if winning_combo:
            self.gameWon.emit(winning_combo)
            QThread.msleep(1000)
            self.finished.emit(f"Winner is {self.game_logic.current_winner}!")
            return True
        elif len(self.game_logic.available_moves()) == 0:
            self.finished.emit("It's a Tie!")
            return True
        return False

    def run_human_vs_ai(self):
        self.game_logic.reset()
        self.boardUpdated.emit(list(self.game_logic.board))
        self.current_player = 'X'
        if self.human_player == 'X':
            self.stateChanged.emit("PLAYING_HUMAN")
        else:
            self.stateChanged.emit("PLAYING_AI")
            QThread.msleep(500)
            self._ai_move()
    
    @Slot()
    def run(self):
        callbacks = {
            'on_progress': self.trainingProgress.emit,
            'on_board_update': self.boardUpdated.emit,
            'on_stats_update': self.statsUpdated.emit,
            'on_log': self.logMessage.emit,
            'on_win': self.gameWon.emit,
            'on_finish': self.finished.emit,
            'get_delay': self.get_delay,
        }
        
        if self.mode == "train":
            self.controller.run_training_session(self.training_episodes, callbacks)
        elif self.mode == "evaluate":
            self.controller.run_evaluation_game(callbacks)
        elif self.mode == "human_vs_ai":
            self.run_human_vs_ai()

class StatWidget(QWidget):
    def __init__(self, name_id):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        self.name_label = QLabel(f"<b>{name_id}</b>")
        self.name_label.setObjectName("header")
        self.value_label = QLabel("0")
        self.value_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.name_label)
        layout.addStretch()
        layout.addWidget(self.value_label)
        self.setLayout(layout)
    
    def set_value(self, value):
        self.value_label.setText(str(value))

class MainWindow(QMainWindow):
    humanMoveSignal = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Tic-Tac-Toe")
        self.setFixedSize(1100, 700)
        self.setWindowIcon(qta.icon('fa5s.brain'))
        
        self.agent_x = DuelingDoubleQLearningAgent('X', epsilon_decay=7000)
        self.agent_o = DuelingDoubleQLearningAgent('O', epsilon_decay=7000)
        self.worker_thread = None
        self.game_worker = None
        
        self.setup_ui()
        self.apply_theme("dark_alliance")
        self.set_application_state("IDLE")

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setOffset(0, 0)
        left_panel.setGraphicsEffect(shadow)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 8)
        self.status_header = QLabel()
        self.status_header.setObjectName("statusHeader")
        self.status_header.setAlignment(Qt.AlignCenter)
        board_bezel = QFrame()
        board_bezel.setObjectName("boardBezel")
        bezel_layout = QVBoxLayout(board_bezel)
        self.board_widget = GameBoardWidget()
        bezel_layout.addWidget(self.board_widget)
        left_layout.addWidget(self.status_header)
        left_layout.addStretch(1)
        left_layout.addWidget(board_bezel, 0, Qt.AlignCenter)
        left_layout.addStretch(1)
        main_layout.addWidget(left_panel, 2)
        
        right_panel = QTabWidget()
        self.controls_tab = self.create_controls_tab()
        self.stats_tab = self.create_stats_tab()
        self.log_tab = self.create_log_tab()
        right_panel.addTab(self.controls_tab, qta.icon('fa5s.gamepad'), "Controls")
        right_panel.addTab(self.stats_tab, qta.icon('fa5s.chart-bar'), "Statistics")
        right_panel.addTab(self.log_tab, qta.icon('fa5s.file-alt'), "Log")
        main_layout.addWidget(right_panel, 1)

        self.setCentralWidget(main_widget)
        self.setStatusBar(QStatusBar(self))

    def create_controls_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 15, 10, 10)
        
        game_group = QGroupBox("Player vs AI")
        game_layout = QVBoxLayout(game_group)
        self.play_as_x_button = QPushButton(qta.icon('fa5s.user'), " Play as X")
        self.play_as_o_button = QPushButton(qta.icon('fa5s.robot'), " Play as O")
        self.play_as_x_button.clicked.connect(lambda: self.start_game("human_vs_ai", human_player='X'))
        self.play_as_o_button.clicked.connect(lambda: self.start_game("human_vs_ai", human_player='O'))
        game_layout.addWidget(self.play_as_x_button)
        game_layout.addWidget(self.play_as_o_button)
        layout.addWidget(game_group)

        eval_group = QGroupBox("AI vs AI")
        eval_layout = QVBoxLayout(eval_group)
        self.eval_button = QPushButton(qta.icon('fa5s.search'), " Run Evaluation Game")
        self.eval_button.clicked.connect(lambda: self.start_game("evaluate"))
        eval_layout.addWidget(self.eval_button)
        layout.addWidget(eval_group)

        train_group = QGroupBox("Training")
        train_layout = QGridLayout(train_group)
        train_layout.addWidget(QLabel("Games:"), 0, 0)
        self.games_spinbox = QSpinBox()
        self.games_spinbox.setRange(100, 1000000)
        self.games_spinbox.setSingleStep(100)
        self.games_spinbox.setValue(10000)
        train_layout.addWidget(self.games_spinbox, 0, 1)
        
        self.train_button = QPushButton(qta.icon('fa5s.dumbbell'), " Start Training")
        self.train_button.clicked.connect(self.toggle_training)
        train_layout.addWidget(self.train_button, 1, 0, 1, 2)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(0)
        speed_layout.addWidget(self.speed_slider)
        train_layout.addLayout(speed_layout, 2, 0, 1, 2)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        train_layout.addWidget(self.progress_bar, 3, 0, 1, 2)
        layout.addWidget(train_group)

        agent_group = QGroupBox("Agent Management")
        agent_layout = QHBoxLayout(agent_group)
        self.save_button = QPushButton(qta.icon('fa5s.save'), " Save")
        self.load_button = QPushButton(qta.icon('fa5s.folder-open'), " Load")
        self.save_button.clicked.connect(self.save_agents)
        self.load_button.clicked.connect(self.load_agents)
        agent_layout.addWidget(self.save_button)
        agent_layout.addWidget(self.load_button)
        layout.addWidget(agent_group)

        theme_group = QGroupBox("Appearance")
        theme_layout = QHBoxLayout(theme_group)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark Alliance", "Light Rebellion"])
        self.theme_combo.currentTextChanged.connect(self.on_theme_change)
        theme_layout.addWidget(QLabel("Theme:"))
        theme_layout.addWidget(self.theme_combo)
        layout.addWidget(theme_group)
        
        layout.addStretch()
        return tab

    def create_stats_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)
        self.stats_widgets = {}
        for symbol in ['X', 'O']:
            group = QGroupBox(f"Agent {symbol} Statistics")
            group_layout = QVBoxLayout(group)
            self.stats_widgets[symbol] = {
                "Games": StatWidget("Total Games"),
                "Wins": StatWidget("Wins"),
                "Losses": StatWidget("Losses"),
                "Ties": StatWidget("Ties"),
                "Epsilon": StatWidget("Epsilon"),
            }
            for widget in self.stats_widgets[symbol].values():
                group_layout.addWidget(widget)
            layout.addWidget(group)
        layout.addStretch()
        return tab
        
    def create_log_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 15, 10, 10)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        return tab

    def set_application_state(self, state, message=None):
        self.current_state = state
        self.progress_bar.setVisible(state == "TRAINING")
        if state == "IDLE":
            self.status_header.setText(message or "Ready")
            self.train_button.setText(" Start Training")
            self.train_button.setIcon(qta.icon('fa5s.dumbbell'))
            for widget in [self.play_as_x_button, self.play_as_o_button, self.eval_button,
                           self.train_button, self.save_button, self.load_button, self.games_spinbox, self.theme_combo]:
                widget.setEnabled(True)
            self.board_widget.set_enabled(False)
        elif state == "TRAINING":
            self.status_header.setText("AI Training in Progress...")
            self.train_button.setText(" Stop Training")
            self.train_button.setIcon(qta.icon('fa5s.stop-circle'))
            for widget in [self.play_as_x_button, self.play_as_o_button, self.eval_button,
                           self.save_button, self.load_button, self.games_spinbox, self.theme_combo]:
                widget.setEnabled(False)
        elif state in ["EVALUATING", "PLAYING_AI", "PLAYING_HUMAN"]:
            self.status_header.setText(message or "Game in progress...")
            self.board_widget.set_enabled(state == "PLAYING_HUMAN")
            for widget in [self.play_as_x_button, self.play_as_o_button, self.eval_button, self.train_button, self.theme_combo]:
                widget.setEnabled(False)
                
    def apply_theme(self, theme_name):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setOffset(0, 0)
        shadow_color = QColor("#1a2530") if theme_name == "dark_alliance" else QColor("#99a3a4")
        shadow_color.setAlphaF(0.5)
        shadow.setColor(shadow_color)
        self.findChild(QFrame, "leftPanel").setGraphicsEffect(shadow)
        self.setStyleSheet(get_stylesheet(theme_name))
        
    def on_theme_change(self, text):
        theme_name = "dark_alliance" if text == "Dark Alliance" else "light_rebellion"
        self.apply_theme(theme_name)

    @Slot(list)
    def update_board(self, board_state):
        self.board_widget.set_board_state(board_state)

    @Slot(object, object)
    def update_stats_display(self, agent_x=None, agent_o=None):
        ax = agent_x or self.agent_x
        ao = agent_o or self.agent_o
        self.stats_widgets['X']["Games"].set_value(f"{ax.episode}")
        self.stats_widgets['X']["Wins"].set_value(f"{ax.wins}")
        self.stats_widgets['X']["Losses"].set_value(f"{ax.losses}")
        self.stats_widgets['X']["Ties"].set_value(f"{ax.ties}")
        self.stats_widgets['X']["Epsilon"].set_value(f"{ax.epsilon:.3f}")
        
        self.stats_widgets['O']["Games"].set_value(f"{ao.episode}")
        self.stats_widgets['O']["Wins"].set_value(f"{ao.wins}")
        self.stats_widgets['O']["Losses"].set_value(f"{ao.losses}")
        self.stats_widgets['O']["Ties"].set_value(f"{ao.ties}")
        self.stats_widgets['O']["Epsilon"].set_value(f"{ao.epsilon:.3f}")

    @Slot(int, int)
    def update_training_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setFormat(f"{current} / {total} Games")

    @Slot(str)
    def log_message(self, message):
        self.log_box.append(message)

    @Slot(list)
    def handle_game_won(self, combination):
        self.board_widget.show_winning_line(combination)
        
    @Slot(str)
    def handle_worker_finished(self, message):
        self.stop_current_worker()
        self.set_application_state("IDLE", message)
        self.update_stats_display()
        
    def stop_current_worker(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.game_worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread, self.game_worker = None, None

    def setup_worker(self):
        self.stop_current_worker()
        self.worker_thread = QThread()
        self.game_worker = GameWorker(self.agent_x, self.agent_o)
        self.game_worker.moveToThread(self.worker_thread)
        
        self.game_worker.boardUpdated.connect(self.update_board)
        self.game_worker.statsUpdated.connect(self.update_stats_display)
        self.game_worker.finished.connect(self.handle_worker_finished)
        self.game_worker.logMessage.connect(self.log_message)
        self.game_worker.stateChanged.connect(self.set_application_state)
        self.game_worker.gameWon.connect(self.handle_game_won)
        self.game_worker.trainingProgress.connect(self.update_training_progress)
        self.speed_slider.valueChanged.connect(self.game_worker.set_delay)
        
        self.humanMoveSignal.connect(self.game_worker.human_made_move)
        self.board_widget.cellClicked.connect(self.humanMoveSignal.emit)
        
        return self.game_worker, self.worker_thread

    def toggle_training(self):
        if self.current_state == "TRAINING":
            self.stop_current_worker()
        else:
            self.log_box.clear()
            self.log_message("Starting new training session...")
            self.set_application_state("TRAINING")
            episodes = self.games_spinbox.value()
            self.update_training_progress(0, episodes)
            worker, thread = self.setup_worker()
            worker.set_mode("train", episodes=episodes)
            thread.started.connect(worker.run)
            thread.start()

    def start_game(self, mode, human_player=None):
        self.log_box.clear()
        self.board_widget.reset()
        state = "EVALUATING" if mode == "evaluate" else "PLAYING_AI"
        self.set_application_state(state, "Starting game...")
        worker, thread = self.setup_worker()
        worker.set_mode(mode, human_player=human_player)
        thread.started.connect(worker.run)
        thread.start()

    def save_agents(self):
        filename_x, _ = QFileDialog.getSaveFileName(self, "Save Agent X", "agent_x.pkl", "Pickle Files (*.pkl)")
        if filename_x: self.agent_x.save(filename_x)
        filename_o, _ = QFileDialog.getSaveFileName(self, "Save Agent O", "agent_o.pkl", "Pickle Files (*.pkl)")
        if filename_o: self.agent_o.save(filename_o)
        if filename_x or filename_o: self.statusBar().showMessage("Agents saved successfully.", 3000)

    def load_agents(self):
        filename_x, _ = QFileDialog.getOpenFileName(self, "Load Agent X", "", "Pickle Files (*.pkl)")
        if filename_x: self.agent_x.load(filename_x)
        filename_o, _ = QFileDialog.getOpenFileName(self, "Load Agent O", "", "Pickle Files (*.pkl)")
        if filename_o: self.agent_o.load(filename_o)
        if filename_x or filename_o:
            self.update_stats_display()
            self.statusBar().showMessage("Agents loaded successfully.", 3000)

    def closeEvent(self, event):
        self.stop_current_worker()
        event.accept()