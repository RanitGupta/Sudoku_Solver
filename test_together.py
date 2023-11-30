# from Sudoku_Alg import solve
# from Sudoku_checker import check_board

from Sudoku_Board1 import image_to_array
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("sudoku_test_3.jpg")

plt.imshow(image, cmap="gray")
plt.show()

board = image_to_array(image)
print(board)