import math
import logging
import time
import os
from typing import List, Tuple
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API key
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("No API key found. Please set GOOGLE_API_KEY in .env file.")

# Logging configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    filename='game_log.txt')

# Configuration and Constants
BOARD_SIZES = [3, 4, 5]  # Supported board sizes
PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '

class TicTacToeAI:
    def __init__(self, board_size: int = 3):
        if board_size not in BOARD_SIZES:
            raise ValueError(f"Board size must be one of {BOARD_SIZES}")
        
        self.BOARD_SIZE = board_size
        self.nodes_evaluated = 0
        self.game_stats = {
            'total_time': 0,
            'iterations': 0,
            'nodes_evaluated': 0
        }
        
        # Secure API key handling
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        except Exception as e:
            logging.error(f"API Configuration Error: {e}")
            raise

    def print_board(self, board: List[List[str]]) -> None:
        """Print board in a formatted manner."""
        for row in board:
            print(' | '.join(row))
            print('-' * (self.BOARD_SIZE * 4 - 1))

    def check_win(self, board: List[List[str]], player: str) -> bool:
        """Check if the specified player has won."""
        # Horizontal and vertical checks
        for i in range(self.BOARD_SIZE):
            if all(board[i][j] == player for j in range(self.BOARD_SIZE)) or \
               all(board[j][i] == player for j in range(self.BOARD_SIZE)):
                return True
        
        # Diagonal checks
        if all(board[i][i] == player for i in range(self.BOARD_SIZE)) or \
           all(board[i][self.BOARD_SIZE-1-i] == player for i in range(self.BOARD_SIZE)):
            return True
        
        return False

    def is_board_full(self, board: List[List[str]]) -> bool:
        """Check if the board is completely filled."""
        return all(board[i][j] != EMPTY 
                   for i in range(self.BOARD_SIZE) 
                   for j in range(self.BOARD_SIZE))

    def minimax(self, board: List[List[str]], depth: int, 
                alpha: float, beta: float, is_maximizing: bool) -> float:
        """Minimax algorithm with Alpha-Beta pruning."""
        self.nodes_evaluated += 1
        
        if self.check_win(board, PLAYER_X):
            return 10 - depth
        if self.check_win(board, PLAYER_O):
            return depth - 10
        if self.is_board_full(board):
            return 0
        
        if is_maximizing:
            max_eval = float('-inf')
            for i in range(self.BOARD_SIZE):
                for j in range(self.BOARD_SIZE):
                    if board[i][j] == EMPTY:
                        board[i][j] = PLAYER_X
                        eval = self.minimax(board, depth + 1, alpha, beta, False)
                        board[i][j] = EMPTY
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(self.BOARD_SIZE):
                for j in range(self.BOARD_SIZE):
                    if board[i][j] == EMPTY:
                        board[i][j] = PLAYER_O
                        eval = self.minimax(board, depth + 1, alpha, beta, True)
                        board[i][j] = EMPTY
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval

    def best_move(self, board: List[List[str]]) -> Tuple[int, int]:
        """Find the best move for Player X using Minimax."""
        best_val = float('-inf')
        move = (-1, -1)
        
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if board[i][j] == EMPTY:
                    board[i][j] = PLAYER_X
                    move_val = self.minimax(board, 0, float('-inf'), float('inf'), False)
                    board[i][j] = EMPTY
                    if move_val > best_val:
                        best_val = move_val
                        move = (i, j)
        return move

    def gemini_move(self, board: List[List[str]]) -> Tuple[int, int]:
        """Get move from Gemini API with fallback strategy and improved parsing."""
        try:
            prompt = f"""
            You are a Tic-Tac-Toe expert playing as O. 
            The current board is:
            {'\n'.join([' '.join(row) for row in board])}
            
            IMPORTANT: Respond ONLY with the move as 'row,col' 
            (zero-indexed). For example: '1,2'
            
            Analyze the board and choose the best move for O.
            """
            
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            
            # Try to extract a valid move
            move_text = response.text.strip()
            
            # Multiple parsing strategies
            try:
                # Try direct parsing
                i, j = map(int, move_text.split(','))
                if 0 <= i < self.BOARD_SIZE and 0 <= j < self.BOARD_SIZE and board[i][j] == EMPTY:
                    return (i, j)
            except:
                # Try extracting numbers from the text
                import re
                numbers = re.findall(r'\d+', move_text)
                if len(numbers) >= 2:
                    i, j = int(numbers[0]), int(numbers[1])
                    if 0 <= i < self.BOARD_SIZE and 0 <= j < self.BOARD_SIZE and board[i][j] == EMPTY:
                        return (i, j)
            
            # Fallback to random move if parsing fails
            return self.random_move(board)

        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            return self.random_move(board)

    def random_move(self, board: List[List[str]]) -> Tuple[int, int]:
        """Fallback strategy: choose a random empty cell."""
        empty_cells = [(i, j) for i in range(self.BOARD_SIZE) 
                       for j in range(self.BOARD_SIZE) if board[i][j] == EMPTY]
        return empty_cells[0] if empty_cells else (-1, -1)

    def plot_board(self, board: List[List[str]]) -> None:
        """Enhanced board visualization."""
        plt.figure(figsize=(6, 6))
        plt.title(f'{self.BOARD_SIZE}x{self.BOARD_SIZE} Tic-Tac-Toe')
        plt.imshow([[0]*self.BOARD_SIZE for _ in range(self.BOARD_SIZE)], cmap='binary', alpha=0.1)
        
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                color = 'red' if board[i][j] == PLAYER_X else 'blue' if board[i][j] == PLAYER_O else 'gray'
                plt.text(j, i, board[i][j], 
                         horizontalalignment='center', 
                         verticalalignment='center', 
                         fontsize=20, color=color)
        
        plt.grid(color='black', linestyle='-', linewidth=2)
        plt.xticks(range(self.BOARD_SIZE))
        plt.yticks(range(self.BOARD_SIZE))
        plt.tight_layout()
        plt.savefig('game_board.png')
        plt.close()

    def play_game(self) -> None:
        """Main game loop with enhanced logging and tracking."""
        board = [[EMPTY] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]
        current_player = PLAYER_X
        game_start_time = time.time()
        
        logging.info(f"Starting {self.BOARD_SIZE}x{self.BOARD_SIZE} Tic-Tac-Toe game")
        
        while True:
            self.print_board(board)
            self.plot_board(board)
            
            if current_player == PLAYER_X:
                i, j = self.best_move(board)
                board[i][j] = PLAYER_X
                logging.info(f"Player X chooses ({i+1}, {j+1})")
            else:
                i, j = self.gemini_move(board)
                board[i][j] = PLAYER_O
                logging.info(f"Player O chooses ({i+1}, {j+1})")
            
            if self.check_win(board, PLAYER_X):
                self.print_board(board)
                logging.info("Player X wins!")
                break
            elif self.check_win(board, PLAYER_O):
                self.print_board(board)
                logging.info("Player O wins!")
                break
            elif self.is_board_full(board):
                self.print_board(board)
                logging.info("Draw!")
                break
            
            current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X
        
        game_end_time = time.time()
        game_duration = game_end_time - game_start_time
        
        logging.info(f"Game Duration: {game_duration:.2f} seconds")
        logging.info(f"Nodes Evaluated: {self.nodes_evaluated}")

def main():
    # Example of playing on different board sizes
    for board_size in BOARD_SIZES:
        try:
            print(f"\n--- Playing {board_size}x{board_size} Game ---")
            game = TicTacToeAI(board_size)
            game.play_game()
        except Exception as e:
            logging.error(f"Game setup failed for {board_size}x{board_size}: {e}")

if __name__ == "__main__":
    main()