import numpy as np

# Given a 2D Numpy Array (matrix) of a unsolved sudoku board, output a 2D Array solution (or null if no solution exists) using backtracking
#   Note that empty squares should be denoted by 0
def solve(board):
    #first iterate over all empty squares
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                #given an empty square, iterate over possible solution values
                for possible_val in valid_nums(board, row, col):
                    board[row][col] = possible_val
                    resulting_board = solve(board)
                    if resulting_board is not None:
                        return resulting_board 
                    else:
                        board[row][col] = 0
                return None
    return board

# Given the Sudoku board and position, return a list of possible legal numbers
def valid_nums(board, row, col):
    possible_vals = []
    #iterate over possible numbers (1-9)
    for i in range(1,10):
        if row_valid(board, row, i) and col_valid(board, col, i) and box_valid(board, row, col, i):
            possible_vals.append(i)
    return possible_vals

# Given a value and position, return if value is valid in its row
def row_valid(board, row, val):
    for i in board[row]:
        if val == i:
            return False
    return True

# Given a value and position, return if value is valid in its column
def col_valid(board, col, val):
    for i in board[:,col]:
        if val == i:
            return False
    return True

# Given a value and position, return if value is valid in its sub-box
def box_valid(board, row, col, val):
    #first find sub-box top left coordinate
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3

    for i in range(9):
        if val == board[box_row + i//3, box_col + i%3]:
            return False
    return True