import pygame
import numpy as np
from environment.envs import ReversiEnv


def human_vs_human():
    # Initialize the environment
    env = ReversiEnv(render_mode=None)  # We'll handle rendering ourselves
    obs, _ = env.reset()

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption('Reversi: Human vs Human')
    font = pygame.font.Font(None, 36)

    def render_board():
        screen.fill((0, 128, 0))  # Green background
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

        # Display current player
        current_player = "White" if env.current_player == 1 else "Black"
        player_text = font.render(
            f"Current Player: {current_player}", True, (255, 255, 255))
        screen.blit(player_text, (10, 10))

        # Display valid moves
        valid_moves = env.get_valid_moves()
        for move in valid_moves:
            pygame.draw.circle(screen, (255, 0, 0),
                               (move[1]*env.square_size + env.square_size//2,
                                move[0]*env.square_size + env.square_size//2),
                               5)

        pygame.display.flip()

    # Game main loop
    running = True
    done = False
    truncated = False

    while running:
        render_board()

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse click position
                x, y = pygame.mouse.get_pos()
                col, row = x // env.square_size, y // env.square_size
                action = row * env.board_size + col

                # Try to execute the player's move
                if env.is_valid_move(row, col):
                    obs, reward, done, truncated, info = env.step(action)
                    print(f"Move made: ({row}, {col})")
                    print(
                        f"Reward: {reward}, Done: {done}, Truncated: {truncated}")
                    print(f"Board state:\n{env.board}")
                else:
                    print("Invalid move! Try again.")

        # Check if the game is over
        if done or truncated:
            render_board()
            winner = env.get_winner()
            if winner == 1:
                result = "White Wins!"
            elif winner == -1:
                result = "Black Wins!"
            else:
                result = "It's a Draw!"

            print(result)
            result_text = font.render(result, True, (255, 255, 255))
            screen.blit(result_text, (env.width // 2 - 100, env.height // 2))
            pygame.display.flip()

            # Wait for a few seconds before ending
            pygame.time.wait(3000)
            running = False

    env.close()
    pygame.quit()


if __name__ == "__main__":
    human_vs_human()
