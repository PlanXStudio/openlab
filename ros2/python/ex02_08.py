"""ex02_08.py"""
class TicTacToc:
    MY = -1
    YOU = +1

    def __init__(self):
        self.board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

    def wins(self, player):
        win_state = [
            [self.board[0][0], self.board[0][1], self.board[0][2]],
            [self.board[1][0], self.board[1][1], self.board[1][2]],
            [self.board[2][0], self.board[2][1], self.board[2][2]],
            [self.board[0][0], self.board[1][0], self.board[2][0]],
            [self.board[0][1], self.board[1][1], self.board[2][1]],
            [self.board[0][2], self.board[1][2], self.board[2][2]],
            [self.board[0][0], self.board[1][1], self.board[2][2]],
            [self.board[2][0], self.board[1][1], self.board[0][2]],
        ]
        return [player, player, player] in win_state

    def game_over(self):
        return self.wins(self.MY) or self.wins(self.YOU)

    def empty_cells(self):
        cells = []

        for r, row in enumerate(self.board):
            for c, cell in enumerate(row):
                if cell == 0:
                    cells.append([r, c])

        return cells

    def valid_move(self, row, column):
        return [row, column] in self.empty_cells()

    def set_move(self, row, column, player):
        if self.valid_move(row, column):
            self.board[row][column] = player
            return True
        else:
            return False

    def show(self):
        str_line = '---------------'

        print('\n' + str_line)
        for row in self.board:
            for cell in row:
                print(f'| {cell} |', end='')
            print('\n' + str_line)

    def trun(self, player):
        if len(self.empty_cells()) == 0 or self.game_over():
            return

        print(f"{player} turn >>>")
        while True:
            row = int(input("Enter of row pos: "))
            column = int(input("Enter of column pos: "))
            if self.set_move(row, column, player):
                break
            print("Bad choice")

        self.show()

def main():
    ttt = TicTacToc()
    while len(ttt.empty_cells()) > 0 and not ttt.game_over():
        try:
            ttt.trun(ttt.MY)
            ttt.trun(ttt.YOU)
        except KeyboardInterrupt:
            exit(1)

    if ttt.wins(ttt.MY):
        print(f"{ttt.MY} win")
    elif ttt.wins(ttt.YOU):
        print(f"{ttt.YOU} win")
    else:
        print("tie")

if __name__ == '__main__':
    main()