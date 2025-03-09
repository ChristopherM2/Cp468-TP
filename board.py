import os
import random
from google import genai
client = genai.Client(api_key='secret')

class Connect4:
    def __init__(self, width=7, height=6, board=None):
        self.width = width
        self.height = height

        if not board:
            self.board = [['+' for _ in range(width)] for _ in range(height)]
        else:
            self.board = board

        self.turn = 'X'

    def __str__(self):
        return '\n'.join([' '.join(row) for row in self.board])

    def play(self, col):
        if self.check_win() is not None:
            return 'Game has finished!'
        for row in range(len(self.board) - 1, -1, -1):
            if self.board[row][col] == '+':
                self.board[row][col] = self.turn
                self.turn = 'O' if self.turn == 'X' else 'X'
                return self
        return 'Invalid move'

    def check_win(self):
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] != '+':
                    if self.check_direction(row, col, 1, 0) or self.check_direction(row, col, 0, 1) or self.check_direction(row, col, 1, 1) or self.check_direction(row, col, 1, -1):
                        return self.board[row][col]
        return None

    def check_direction(self, row, col, rowdir, coldir):
        for i in range(1, 4):
            if row + i * rowdir >= len(self.board) or row + i * rowdir < 0 or col + i * coldir >= len(self.board[0]) or col + i * coldir < 0 or self.board[row + i * rowdir][col + i * coldir] != self.board[row][col]:
                return False
        return True

    def minimax(self, depth=6, maximizing=True):
        valid_moves = self.get_valid_moves()
        if depth == 0 or self.check_win() is not None:
            return None, self.evaluate()

        if maximizing:
            max_eval = -float('inf')
            best_move = random.choice(valid_moves)
            for move in valid_moves:
                new_board = self.simulate_move(move, 'X')
                _, eval_score = new_board.minimax(depth - 1, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return best_move, max_eval
        else:
            min_eval = float('inf')
            best_move = random.choice(valid_moves)
            for move in valid_moves:
                new_board = self.simulate_move(move, 'O')
                _, eval_score = new_board.minimax(depth - 1, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
            return best_move, min_eval

    def alphabeta(self, depth=4, alpha=-float('inf'), beta=float('inf'), maximizing=True):
        valid_moves = self.get_valid_moves()
        if depth == 0 or self.check_win() is not None:
            return None, self.evaluate()

        if maximizing:
            max_eval = -float('inf')
            best_move = random.choice(valid_moves)
            for move in valid_moves:
                new_board = self.simulate_move(move, 'X')
                _, eval_score = new_board.alphabeta(depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return best_move, max_eval
        else:
            min_eval = float('inf')
            best_move = random.choice(valid_moves)
            for move in valid_moves:
                new_board = self.simulate_move(move, 'O')
                _, eval_score = new_board.alphabeta(depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return best_move, min_eval

    def evaluate(self):
        score = 0


        center_col = self.width // 2
        center_count = sum([1 for row in range(self.height) if self.board[row][center_col] == 'X'])
        score += center_count * 3


        for row in range(self.height):
            for col in range(self.width):
                if self.board[row][col] == 'X':
                    score += self.score_position(row, col, 'X')
                elif self.board[row][col] == 'O':
                    score -= self.score_position(row, col, 'O')

        return score

    def score_position(self, row, col, player):
        score = 0

        if col <= self.width - 4:
            window = [self.board[row][col + i] for i in range(4)]
            score += self.evaluate_window(window, player)

        if row <= self.height - 4:
            window = [self.board[row + i][col] for i in range(4)]
            score += self.evaluate_window(window, player)

        if row <= self.height - 4 and col <= self.width - 4:
            window = [self.board[row + i][col + i] for i in range(4)]
            score += self.evaluate_window(window, player)

        if row <= self.height - 4 and col >= 3:
            window = [self.board[row + i][col - i] for i in range(4)]
            score += self.evaluate_window(window, player)

        return score

    def evaluate_window(self, window, player):
        score = 0
        opponent = 'O' if player == 'X' else 'X'

        if window.count(player) == 4:  # Win
            score += 100
        elif window.count(player) == 3 and window.count('+') == 1:
            score += 5
        elif window.count(player) == 2 and window.count('+') == 2:
            score += 2

        if window.count(opponent) == 3 and window.count('+') == 1:  # Block opponent
            score -= 50

        return score

    def get_valid_moves(self):
        return [col for col in range(self.width) if self.board[0][col] == '+']

    def simulate_move(self, col, player):
        new_board = [row[:] for row in self.board]
        for row in range(self.height - 1, -1, -1):
            if new_board[row][col] == '+':
                new_board[row][col] = player
                break
        return Connect4(self.width, self.height, new_board)
    def gemini(self):
        prompt = "Following will be a 2d list representing a game state of connect 4. You are player O, what move would you like to make. Please respond with just a number representing which columun." + self.__str__()
        print(prompt)
        move = client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text
        print(move)
        self.play(int(move)-1)

    def ai_move(self):
        move, _ = self.alphabeta()
        if move is not None:
            self.play(move)



