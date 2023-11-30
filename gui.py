from tkinter import *
from tkinter import filedialog
import cv2
import numpy as  np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# Globals
img = None # Sudoku original image

def openImage():
    #get image path
    global img 
    img_path = filedialog.askopenfilename()

    #save image as cv2 object
    img = cv2.imread(img_path)

#open main app window
app = Tk()

#Create a button to access user sudoku file
open_image_button = Button(app, text='Select Image of Sudoku Puzzle', command=openImage)
open_image_button.pack()

#Display Image inside label
image_label = Label(app, bg='black')
image_label.pack()

while True:
    if img is not None:
        PIL_img = ImageTk.PhotoImage(Image.fromarray(img))
        image_label['image'] = PIL_img
    app.update()
