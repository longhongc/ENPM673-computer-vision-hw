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
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

# Do more erosion to make coin smaller
eroded = erosion(thresh, 25); 
cv.imwrite('./pictures/Q1_erosion_for_counting.jpg', eroded)
white = np.where(eroded == 255)

# Connected Component Labeling
connected_class = np.zeros(gray.shape[:2])

# These four neighbors is enough because search algorithm is row wise
neighbors = [(-1, -1), (-1, 0), (-1, 1),
             ( 0, -1)]

class_count = 0
equivalency_lists = [set()] 

total_white = len(white[0])
for i in range(0, total_white):
    row, col = white[0][i], white[1][i]
    for neighbor in neighbors:
        dr, dc = neighbor
        neighbor_class = connected_class[row+dr, col+dc]
        current_class = connected_class[row, col]
        if neighbor_class != 0:
            if (current_class != 0) and (current_class != neighbor_class):
                # Add neighbor class and current class to equivalency_lists
                equivalency_lists[current_class].add(neighbor_class)
                equivalency_lists[neighbor_class].add(current_class)
            else:
                connected_class[row, col] = neighbor_class
    # No neighbor with class
    if connected_class[row, col] == 0:
       # Add a new class
       class_count += 1
       equivalency_lists.append(set({class_count}))
       connected_class[row, col] = class_count

classes = []
for i in range(1, class_count+1):
    min_eq = min(equivalency_lists[i])
    # Find the minimum representative in equivalency_lists 
    while True:
        if min(equivalency_lists[min_eq]) == min_eq:
            # min_eq is the minimum representative 
            # when the minimum class in its equivalency_lists is itself
            classes.append(min_eq)
            break
        min_eq = min(equivalency_lists[min_eq])

print("coins amount: ", len(classes))

