import pygame
import numpy as np
from stable_baselines3 import PPO
from environment.envs import ReversiEnv


def human_vs_ai():
    # 初始化環境
    env = ReversiEnv(render_mode=None)  # 我們將自己處理渲染
    obs, _ = env.reset()

    # 加載訓練好的模型
    model = PPO.load("models/PPO")

    # 初始化Pygame
    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption('Reversi: Human vs AI')
    font = pygame.font.Font(None, 36)

    def render_board():
        screen.fill((0, 128, 0))  # 綠色背景
        for i in range(env.board_size):
            for j in range(env.board_size):
                pygame.draw.rect(screen, (0, 0, 0),
                                 (j*env.square_size, i*env.square_size,
                                  env.square_size, env.square_size), 1)
                if env.board[i, j] != 0:
                    color = (255, 255, 255) if env.board[i, j] == 1 else (
                        0, 0, 0)
                    pygame.draw.circle(screen, color,
                                       (j*env.square_size + env.square_size//2,
                                        i*env.square_size + env.square_size//2),
                                       env.square_size//2 - 5)

        # 顯示當前玩家
        current_player = "Human" if human_turn else "AI"
        player_text = font.render(
            f"Current Player: {current_player}", True, (255, 255, 255))
        screen.blit(player_text, (10, 10))

        # 顯示可行的移動
        valid_moves = env.get_valid_moves()
        for move in valid_moves:
            pygame.draw.circle(screen, (255, 0, 0),
                               (move[1]*env.square_size + env.square_size//2,
                                move[0]*env.square_size + env.square_size//2),
                               5)

        pygame.display.flip()

    def ai_turn():
        nonlocal obs, done, truncated
        print("AI's turn")
        valid_moves = env.get_valid_moves()
        print(f"Valid moves: {valid_moves}")

        if not valid_moves:
            print("AI has no valid moves. Skipping turn.")
            return False

        action, _states = model.predict(obs, deterministic=True)
        row, col = divmod(action, env.board_size)
        print(f"AI chose action: {action} (row: {row}, col: {col})")

        if (row, col) not in valid_moves:
            print(
                f"AI's move ({row}, {col}) is not valid. Choosing random valid move.")
            action = env.board_size * valid_moves[0][0] + valid_moves[0][1]

        obs, reward, done, truncated, info = env.step(action)
        print(f"Action taken: {action}")
        print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")
        print(f"Board state:\n{env.board}")

        render_board()
        return True

    # 遊戲主循環
    running = True
    human_turn = True  # 人類玩家先手
    done = False
    truncated = False

    while running:
        render_board()

        # 處理Pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and human_turn:
                # 獲取鼠標點擊位置
                x, y = pygame.mouse.get_pos()
                col, row = x // env.square_size, y // env.square_size
                action = row * env.board_size + col

                # 嘗試執行人類玩家的移動
                if env.is_valid_move(row, col):
                    obs, reward, done, truncated, info = env.step(action)
                    human_turn = False
                else:
                    print("Invalid move! Try again.")

        if not human_turn and not (done or truncated):
            ai_moved = ai_turn()
            if ai_moved:
                human_turn = True
            else:
                # 如果AI沒有有效移動，檢查人類是否有有效移動
                if not env.get_valid_moves():
                    done = True  # 如果雙方都沒有有效移動，遊戲結束

        # 檢查遊戲是否結束
        if done or truncated:
            render_board()
            winner = env.get_winner()
            if winner == 1:
                result = "Human Wins!"
            elif winner == -1:
                result = "AI Wins!"
            else:
                result = "It's a Draw!"

            print(result)
            result_text = font.render(result, True, (255, 255, 255))
            screen.blit(result_text, (env.width // 2 - 100, env.height // 2))
            pygame.display.flip()

            # 等待幾秒後結束
            pygame.time.wait(3000)
            running = False

    env.close()
    pygame.quit()


if __name__ == "__main__":
    human_vs_ai()
