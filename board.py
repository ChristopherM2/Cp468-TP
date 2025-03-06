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
        for row in range(len(self.board)-1, -1, -1):
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
            if row + i*rowdir >= len(self.board) or row + i*rowdir < 0 or col + i*coldir >= len(self.board[0]) or col + i*coldir < 0 or self.board[row + i*rowdir][col + i*coldir] != self.board[row][col]:
                return False
        return True












