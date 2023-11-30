from tkinter import *
from tkinter import filedialog
import cv2
import numpy as  np
import pandas as pd
import sys
from PIL import Image, ImageTk


from Sudoku_Alg import solve
from Sudoku_checker import check_board
from Sudoku_Board1 import image_to_array

# Globals
img = None # Sudoku original image
solution = None # Sudoku Solution Matrix
newImg = False # If a new Image has been loaded
newSolution = False #If a new Solution has been calculated

def openImage():
    #get image path
    global img 
    img_path = filedialog.askopenfilename()

    #save image as cv2 object
    img = cv2.imread(img_path)

    # update flag
    global newImg
    newImg = True

def solveImg():
    global solution
    if not newImg:
        return
    board = image_to_array(img)
    solution = solve(np.array(board))

    global newSolution
    newSolution = True
    return

#create main app window
app = Tk()
buttonFrame = Frame(app)
imageFrame = Frame(app)
buttonFrame.pack(side=TOP)
imageFrame.pack(side=BOTTOM, fill=BOTH, expand=True)

#Create the buttons
open_image_button = Button(app, text='Select Image of Sudoku Puzzle', command=openImage)
solve_button = Button(app, text='Solve', command=solveImg)
open_image_button.pack(in_=buttonFrame, side=LEFT)
solve_button.pack(in_=buttonFrame, side=LEFT)

#Display Image inside label
image_label = Label(app, bg='black')
image_label.pack(in_=imageFrame, side=LEFT, fill=BOTH, expand=TRUE)

#Display Solution Matrix
solution_label = Label(app, bg='white')
solution_label.pack(in_=imageFrame, side=LEFT, fill=BOTH, expand=TRUE)

while True:
    if img is not None:
        PIL_img = ImageTk.PhotoImage(Image.fromarray(img))
        image_label['image'] = PIL_img
    if newSolution and newImg:
        solution_label['text'] = np.matrix(solution)
        print(np.matrix(solution))
        newImg = False
    app.update()
