import numpy as np

# Given a solved Sudoku board (2D Numpy Array), return if it is a valid sudoku board
def check_board(board):

    for i in range(9):
        row = board[i]
        col = board[:,i]
        box = board[(i//3)*3:(i//3)*3+3, (i%3)*3:(i%3)*3+3].flatten()
        for val in range(1,10):
            if row.tolist().count(val) != 1 or col.tolist().count(val) != 1 or box.tolist().count(val) != 1:
                return False
    return True