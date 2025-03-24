import math
from google import genai

import pathlib
client = genai.Client(api_key='AIzaSyBDt7xyy-szMGlJ15tYUb3T7MrjV3yymGA')
# Constants for the game
PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '

# Function to print the Tic-Tac-Toe board
def print_board(board):
    for row in board:
        print(' | '.join(row))
        print('-' * 10)
def tostr(board):
    meow = ''
    for row in board:
        meow += 'Row:'
        meow+=' | '.join(row)
        meow+='-' * 10
    return meow
# Function to check if the current player has won
def check_win(board, player):
    # Check rows, columns and diagonals
    for i in range(3):
        if all([board[i][j] == player for j in range(3)]) or all([board[j][i] == player for j in range(3)]):
            return True
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True
    return False

# Function to check if the board is full
def is_full(board):
    return all([board[i][j] != EMPTY for i in range(3) for j in range(3)])

# Minimax algorithm with Alpha-Beta Pruning :fire:
def minimax(board, depth, alpha, beta, is_maximizing):
    if check_win(board, PLAYER_X):
        return 10 - depth
    if check_win(board, PLAYER_O):
        return depth - 10
    if is_full(board):
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    board[i][j] = PLAYER_X
                    eval = minimax(board, depth + 1, alpha, beta, False)
                    board[i][j] = EMPTY
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    board[i][j] = PLAYER_O
                    eval = minimax(board, depth + 1, alpha, beta, True)
                    board[i][j] = EMPTY
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval


def best_move(board):
    best_val = -math.inf
    move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                board[i][j] = PLAYER_X
                move_val = minimax(board, 0, -math.inf, math.inf, False)
                board[i][j] = EMPTY
                if move_val > best_val:
                    best_val = move_val
                    move = (i, j)
    return move
def gemini_move(board):
    prompt = "You are a tic-tac-toe bot expert. Following this you will see the current board state, you are player O, return the best move in the format of {row,col} do not say anything other than {row,col}"

    prompt = prompt + tostr(board)
    media = pathlib.Path(__file__).parents[1] / "TP" #using example in https://github.com/google-gemini/api-examples/blob/7ab021b5343e58303522449cb1b8746d1b7207b4/python/files.py#L136-L144
    file = client.files.upload(file=media / "tictactoe.pdf")
    meow = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[prompt,file],
    )
    print(prompt)
    print('\n')
    print(meow.text)
    i = meow.text.strip().split(',')[0]
    i = i.replace(' ', '')
    i = i.replace('{', '')
    j = meow.text.strip().split(',')[1]
    j = j.replace(' ', '')
    j = j.replace('}', '')
    print(i)
    print(j)
    return int(i),int(j)
# Main game loop

def play_game():
    board = [[EMPTY] * 3 for _ in range(3)]
    current_player = PLAYER_X
    print('starting')
    while True:
        print_board(board)
        if current_player == PLAYER_X:
            i, j = best_move(board)
            board[i][j] = PLAYER_X
            print(f"Player X chooses ({i+1}, {j+1})\n")
        else:
            i, j = gemini_move(board)
            board[i][j] = PLAYER_O
            print(f"Player O chooses ({i+1}, {j+1})\n")

        if check_win(board, PLAYER_X):
            print_board(board)
            print("Player X wins!")
            break
        elif check_win(board, PLAYER_O):
            print_board(board)
            print("Player O wins!")
            break
        elif is_full(board):
            print_board(board)
            print("draw!")
            break

        current_player = PLAYER_X if current_player == PLAYER_O else PLAYER_O

if __name__ == "__main__":
    print('meow')
    play_game()
