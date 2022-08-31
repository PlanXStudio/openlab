"""ex02_05.py"""
MY = -1
YOU = +1

board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

def wins(player):
    win_state = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]],
    ]
    return [player, player, player] in win_state

def game_over():
    return wins(MY) or wins(YOU)

def empty_cells():
    cells = []

    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell == 0:
                cells.append([r, c])

    return cells

def valid_move(row, column):
    return [row, column] in empty_cells()

def set_move(row, column, player):
    if valid_move(row, column):
        board[row][column] = player
        return True
    else:
        return False

def show():
    str_line = '---------------'

    print('\n' + str_line)
    for row in board:
        for cell in row:
            print(f'| {cell} |', end='')
        print('\n' + str_line)

def trun(player):
    if len(empty_cells()) == 0 or game_over():
        return

    print(f"{player} turn >>>")
    while True:
        row = int(input("Enter of row pos: "))
        column = int(input("Enter of column pos: "))
        if set_move(row, column, player):
            break
        print("Bad choice")

    show()

def main():
    show()
    while len(empty_cells()) > 0 and not game_over():
        try:
            trun(MY)
            trun(YOU)
        except KeyboardInterrupt:
            exit(1)

    if wins(MY):
        print(f"{MY} win")
    elif wins(YOU):
        print(f"{YOU} win")
    else:
        print("tie")

if __name__ == '__main__':
    main()