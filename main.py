import pygame
from board import Connect4
from pygame import *



def draw_text(screen, text, size, x, y):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    screen.blit(text_surface, text_rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((1000, 600))
    pygame.display.set_caption('meows')
    clock = pygame.time.Clock()
    choice = None

    board = Connect4(7, 6)
    print(board)

    while True:
        screen.fill((0, 0, 0))

        # Draw the game board
        for row in range(6):
            for col in range(7):
                pygame.draw.rect(screen, (255, 255, 255), (col * 100, row * 100, 100, 100))
                pygame.draw.circle(screen,
                                   (255, 0, 0) if board.board[row][col] == 'X' else (0, 0, 255) if board.board[row][
                                                                                                       col] == 'O' else (
                                   0, 0, 0), (col * 100 + 50, row * 100 + 50), 40)

        # Draw the difficulty selection buttons
        draw_text(screen, 'What Ai would you like to use', 30, 850, 50)
        draw_text(screen, '1. Minimax', 30, 850, 150)
        draw_text(screen, '2. Alpha-Beta', 30, 850, 250)
        draw_text(screen, '3. Gemini', 30, 850, 350)

        draw_text(screen, f'Choice: {choice}', 30, 850, 450)


        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            if event.type == MOUSEBUTTONDOWN:
                col = event.pos[0] // 100
                if col < 7:
                    board.play(col)
                    print('\n')
                    print(board)
                    if not choice:
                        choice='Minimax'
                    if choice == 'Minimax':
                        move, _ = board.minimax()
                        if move is not None:
                            board.play(move)
                            print(board)
                    elif choice == 'Alpha-Beta':
                        move, _ = board.alphabeta()
                        if move is not None:
                            board.play(move)
                    elif choice == 'Gemini':
                        board.gemini()

                else:
                    if 800 < event.pos[0] < 1000:
                        if 150 < event.pos[1] < 200:
                            choice = 'Minimax'
                        elif 250 < event.pos[1] < 300:
                            choice = 'Alpha-Beta'
                        elif 350 < event.pos[1] < 400:
                            choice = 'Gemini'
                        if choice:
                            print(f'Choice set to {choice}')

        pygame.display.flip()
        clock.tick(30)

        if board.check_win() is not None:
            print(f'{board.check_win()} wins!')
            break


if __name__ == '__main__':
    main()