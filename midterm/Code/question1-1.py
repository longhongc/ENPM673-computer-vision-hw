import cv2 as cv
import numpy as np

def dilation(frame, size):
    mask = np.zeros((size*2, size*2), dtype='uint8')
    cv.circle(mask,(mask.shape[1]//2, mask.shape[0]//2), size, 255, -1)
    dilated = cv.dilate(frame, mask)

    return dilated

def erosion(frame, size):
    mask = np.zeros((size*2, size*2), dtype='uint8')
    cv.circle(mask,(mask.shape[1]//2, mask.shape[0]//2), size, 255, -1)
    eroded = cv.erode(frame, mask)

    return eroded

 
    
img = cv.imread('./pictures/Q1image.png')

ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
cv.imshow('Coins', thresh)


# Opening
eroded = erosion(thresh, 15); 
dilated = dilation(eroded, 15); 

cv.imshow('Seperated', dilated)

cv.imwrite('./pictures/Q1_seperated_coins.jpg', dilated)

cv.waitKey(0)
