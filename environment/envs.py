import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class ReversiEnv(gym.Env):
    def __init__(self, board_size=8, render_mode=None):
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode
        self.square_size = 80
        self.width = self.height = self.board_size * self.square_size

        # 定義動作空間和觀察空間
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)

        # 如果需要渲染,初始化Pygame
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Reversi')
            self.font = pygame.font.Font(None, 36)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 重置遊戲板到初始狀態
        self.board = np.zeros(
            (self.board_size, self.board_size), dtype=np.int8)
        center = self.board_size // 2
        self.board[center-1:center+1, center -
                   1:center+1] = np.array([[1, -1], [-1, 1]])
        self.current_player = 1
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        # 執行一步動作
        row, col = divmod(action, self.board_size)
        new_board, flipped, self.done, info = self.make_move(row, col)

        # 計算獎勵
        reward = flipped if self.current_player == 1 else -flipped

        if self.done:
            winner = self.get_winner()
            if winner == 1:
                reward += 100
            elif winner == -1:
                reward -= 100

        return new_board, reward, self.done, False, info  # Added terminated flag

    def get_valid_moves(self):
        # 獲取所有合法移動
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves

    def is_valid_move(self, row, col):
        # 檢查移動是否合法
        if self.board[row, col] != 0:
            return False

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            if self.would_flip(row, col, dr, dc):
                return True
        return False

    def would_flip(self, row, col, dr, dc):
        # 檢查是否會翻轉對手的棋子
        r, c = row + dr, col + dc
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return False
        if self.board[r, c] != -self.current_player:
            return False
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if self.board[r, c] == 0:
                return False
            if self.board[r, c] == self.current_player:
                return True
            r, c = r + dr, c + dc
        return False

    def make_move(self, row, col):
        # 執行移動並翻轉棋子
        if not self.is_valid_move(row, col):
            return self.board.copy(), 0, self.done, {"error": "Invalid move"}

        flipped = 0
        self.board[row, col] = self.current_player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            if self.would_flip(row, col, dr, dc):
                r, c = row + dr, col + dc
                while self.board[r, c] != self.current_player:
                    self.board[r, c] = self.current_player
                    flipped += 1
                    r, c = r + dr, c + dc

        self.current_player = -self.current_player
        if not self.get_valid_moves():
            self.current_player = -self.current_player
            if not self.get_valid_moves():
                self.done = True

        return self.board.copy(), flipped, self.done, {}

    def get_winner(self):
        # 獲取獲勝者
        if not self.done:
            return None
        player_sum = np.sum(self.board)
        if player_sum > 0:
            return 1
        elif player_sum < 0:
            return -1
        else:
            return 0

    def render(self):
        # 渲染遊戲狀態
        if self.render_mode != "human":
            return

        self.screen.fill((0, 128, 0))  # 綠色背景

        for i in range(self.board_size):
            for j in range(self.board_size):
                pygame.draw.rect(self.screen, (0, 0, 0),
                                 (i*self.square_size, j*self.square_size,
                                  self.square_size, self.square_size), 1)

                if self.board[i, j] != 0:
                    color = (255, 255, 255) if self.board[i, j] == 1 else (
                        0, 0, 0)
                    pygame.draw.circle(self.screen, color,
                                       (i*self.square_size + self.square_size//2,
                                        j*self.square_size + self.square_size//2),
                                       self.square_size//2 - 5)

        player_text = self.font.render(f"Current Player: {'White' if self.current_player == 1 else 'Black'}",
                                       True, (255, 255, 255))
        self.screen.blit(player_text, (10, 10))

        pygame.display.flip()

    def close(self):
        # 關閉環境
        if self.render_mode == "human":
            pygame.quit()
