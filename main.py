import math
import logging
import time
import os
import re
from typing import List, Tuple, Callable
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='game_log.txt',
    filemode='w'
)

# Configuration and Constants
BOARD_SIZES = [3, 4, 5]  # Supported board sizes
PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '

class TicTacToeAI:
    def __init__(self, board_size: int = 3, max_depth: int = 5):
        """
        Initialize Tic-Tac-Toe AI with configurable board size and max search depth
        
        Args:
            board_size (int): Size of the game board (3, 4, or 5)
            max_depth (int): Maximum depth for minimax search to prevent excessive computation
        """
        if board_size not in BOARD_SIZES:
            raise ValueError(f"Board size must be one of {BOARD_SIZES}")
        
        self.BOARD_SIZE = board_size
        self.MAX_DEPTH = max_depth
        self.nodes_evaluated = 0
        
        # Secure API key handling
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            try:
                genai.configure(api_key=api_key)
            except Exception as e:
                logging.error(f"API Configuration Error: {e}")
                raise
        else:
            logging.warning("No Gemini API key found. Gemini-based moves will use random strategy.")

    def print_board(self, board: List[List[str]]) -> None:
        """Print board in a formatted manner."""
        print(f"\n{self.BOARD_SIZE}x{self.BOARD_SIZE} Board:")
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
        """Minimax algorithm with Alpha-Beta pruning and depth limitation"""
        self.nodes_evaluated += 1
        
        # Terminal state checks
        if depth >= self.MAX_DEPTH:
            return 0
        
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
        """Find the best move for Player X using Minimax with enhanced logging"""
        logging.info(f"Calculating best move for board size {self.BOARD_SIZE}")
        
        best_val = float('-inf')
        move = (-1, -1)
        moves_considered = 0
        
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if board[i][j] == EMPTY:
                    board[i][j] = PLAYER_X
                    move_val = self.minimax(board, 0, float('-inf'), float('inf'), False)
                    board[i][j] = EMPTY
                    
                    logging.debug(f"Move ({i},{j}) evaluated with value: {move_val}")
                    moves_considered += 1
                    
                    if move_val > best_val:
                        best_val = move_val
                        move = (i, j)
        
        logging.info(f"Total moves considered: {moves_considered}")
        
        if move == (-1, -1):
            logging.error("No valid move found!")
            raise ValueError("Unable to find a valid move")
        
        return move

    def gemini_move(self, board: List[List[str]]) -> Tuple[int, int]:
        """Enhanced Gemini move generation with robust error handling"""
        logging.info("Attempting to get move from Gemini API")
        
        # Fallback if no API key
        if not os.getenv('GEMINI_API_KEY'):
            return self.random_move(board)
        
        try:
            # More detailed board representation
            board_repr = '\n'.join([' '.join(row).replace(' ', '_') for row in board])
            
            prompt = f"""
            Tic-Tac-Toe board (size: {self.BOARD_SIZE}x{self.BOARD_SIZE}):
            {board_repr}

            You are playing as O. Choose an empty cell. 
            Respond STRICTLY in format: 'row,column' (zero-indexed)
            Empty cells are represented by '_'
            """
            
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            
            # Multiple parsing strategies with logging
            move_candidates = [
                # Direct comma-separated parsing
                self._parse_move_direct(response.text, board),
                # Regex-based parsing
                self._parse_move_regex(response.text, board),
                # Random move as final fallback
                self.random_move(board)
            ]
            
            # Return first valid move
            for move in move_candidates:
                if move != (-1, -1) and board[move[0]][move[1]] == EMPTY:
                    logging.info(f"Gemini selected move: {move}")
                    return move
            
            raise ValueError("No valid move found")
        
        except Exception as e:
            logging.error(f"Gemini move generation failed: {e}")
            return self.random_move(board)

    def _parse_move_direct(self, text: str, board: List[List[str]]) -> Tuple[int, int]:
        """Parse move directly from comma-separated coordinates"""
        try:
            parts = text.strip().split(',')
            if len(parts) == 2:
                i, j = map(int, parts)
                if 0 <= i < self.BOARD_SIZE and 0 <= j < self.BOARD_SIZE:
                    return (i, j)
        except Exception as e:
            logging.warning(f"Direct parsing failed: {e}")
        return (-1, -1)

    def _parse_move_regex(self, text: str, board: List[List[str]]) -> Tuple[int, int]:
        """Parse move using regex with flexible matching"""
        try:
            numbers = re.findall(r'\b\d+\b', text)
            if len(numbers) >= 2:
                i, j = int(numbers[0]), int(numbers[1])
                if 0 <= i < self.BOARD_SIZE and 0 <= j < self.BOARD_SIZE:
                    return (i, j)
        except Exception as e:
            logging.warning(f"Regex parsing failed: {e}")
        return (-1, -1)

    def random_move(self, board: List[List[str]]) -> Tuple[int, int]:
        """Fallback strategy: choose a random empty cell"""
        empty_cells = [(i, j) for i in range(self.BOARD_SIZE) 
                       for j in range(self.BOARD_SIZE) if board[i][j] == EMPTY]
        return empty_cells[0] if empty_cells else (-1, -1)

    def plot_board(self, board: List[List[str]]) -> None:
        """Enhanced board visualization"""
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

    def play_game(self, player_x_strategy: str = 'minimax', 
                  player_o_strategy: str = 'gemini') -> dict:
        """
        Enhanced game play with configurable strategies and result tracking
        
        Args:
            player_x_strategy (str): Strategy for X ('minimax', 'random', 'gemini')
            player_o_strategy (str): Strategy for O ('minimax', 'random', 'gemini')
        
        Returns:
            dict: Game results and statistics
        """
        board = [[EMPTY] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]
        current_player = PLAYER_X
        game_start_time = time.time()
        
        logging.info(f"Starting {self.BOARD_SIZE}x{self.BOARD_SIZE} Tic-Tac-Toe")
        logging.info(f"Player X Strategy: {player_x_strategy}")
        logging.info(f"Player O Strategy: {player_o_strategy}")
        
        max_moves = self.BOARD_SIZE * self.BOARD_SIZE
        move_count = 0
        game_result = {
            'winner': None,
            'result': 'Draw',
            'board_size': self.BOARD_SIZE,
            'x_strategy': player_x_strategy,
            'o_strategy': player_o_strategy
        }
        
        # Strategy mapping
        strategies = {
            'minimax': self.best_move,
            'random': self.random_move,
            'gemini': self.gemini_move
        }
        
        while move_count < max_moves:
            self.print_board(board)
            self.plot_board(board)
            
            # Determine move strategy based on current player
            if current_player == PLAYER_X:
                move_func = strategies.get(player_x_strategy, self.best_move)
                i, j = move_func(board)
                board[i][j] = PLAYER_X
                logging.info(f"Player X ({player_x_strategy}) chooses ({i+1}, {j+1})")
            else:
                move_func = strategies.get(player_o_strategy, self.gemini_move)
                i, j = move_func(board)
                board[i][j] = PLAYER_O
                logging.info(f"Player O ({player_o_strategy}) chooses ({i+1}, {j+1})")
            
            move_count += 1
            
            if self.check_win(board, PLAYER_X):
                self.print_board(board)
                logging.info(f"Player X ({player_x_strategy}) wins!")
                game_result['winner'] = 'X'
                game_result['result'] = 'X Wins'
                break
            elif self.check_win(board, PLAYER_O):
                self.print_board(board)
                logging.info(f"Player O ({player_o_strategy}) wins!")
                game_result['winner'] = 'O'
                game_result['result'] = 'O Wins'
                break
            elif self.is_board_full(board):
                self.print_board(board)
                logging.info("Draw!")
                break
            
            current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X
        
        game_end_time = time.time()
        game_duration = game_end_time - game_start_time
        
        # Update game result statistics
        game_result.update({
            'duration': game_duration,
            'total_moves': move_count,
            'nodes_evaluated': self.nodes_evaluated
        })
        
        logging.info(f"Game Duration: {game_duration:.2f} seconds")
        logging.info(f"Total Moves: {move_count}")
        logging.info(f"Nodes Evaluated: {self.nodes_evaluated}")
        
        return game_result

def main():
    """
    Demonstrate various strategy combinations across different board sizes
    """
    strategies = [
        ('minimax', 'random'),
        ('minimax', 'gemini'),
        ('random', 'gemini')
    ]
    
    results = []
    
    for board_size in BOARD_SIZES:
        for x_strategy, o_strategy in strategies:
            print(f"\n--- {board_size}x{board_size} Game: X={x_strategy}, O={o_strategy} ---")
            
            try:
                game = TicTacToeAI(board_size)
                game_result = game.play_game(
                    player_x_strategy=x_strategy, 
                    player_o_strategy=o_strategy
                )
                results.append(game_result)
            except Exception as e:
                logging.error(f"Game failed for {board_size}x{board_size}: {e}")
    
    print("\n--- Game Results Summary ---")
    for result in results:
        print(f"Board Size: {result['board_size']}, "
              f"Strategies: X={result['x_strategy']}, O={result['o_strategy']}, "
              f"Result: {result['result']}")

if __name__ == "__main__":
    main()