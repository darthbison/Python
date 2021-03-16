import cv2 
import numpy as np 
 
img = cv2.imread('/home/darthbison/code/Toolbelt/Python/opencv/examples/Chapter02/images/train_input.png', 0) 
 
# equalize the histogram of the input image 
histeq = cv2.equalizeHist(img) 
 
cv2.imshow('Input', img) 
cv2.imshow('Histogram equalized', histeq) 
cv2.waitKey(0) 