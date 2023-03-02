import cv2 as cv
import numpy as np

img = cv.imread('./pictures/Q4image.png')

# Calculate distance between vectors in matrix and a given vector
def cal_distance(frame, mean):
    return np.sqrt(np.sum((frame-mean)**2, axis=-1))

def kmeans(frame, iteration=1): 
    h, w = frame.shape[:2]

    # Set init means to blue, green, red, black 
    means = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]]
    k = len(means)

    count = 0
    while count < iteration: 
        dists_list = [] 
        for mean in means:
            dist = cal_distance(frame, mean)
            dists_list.append(dist.reshape((h, w, 1)))

        # Stack every dists with means 
        dists = np.concatenate(dists_list, axis=-1)
        # Find the smallest dists
        classes = np.argmin(dists, axis=-1)

        means = []
        for i in range(0, k):
            # Get all pixels that are class i
            class_i = frame[np.where(classes == i)]
            if len(class_i) == 0:
                break
            means.append(np.sum(class_i, axis=0) / len(class_i))

        count += 1

    return np.uint(means), classes

# Run 50 iterations
means, classes = kmeans(img, 50)
k = len(means)

kmeans_img = img.copy()

for i in range(0, k): 
    kmeans_img[np.where(classes == i)] = means[i]

cv.imshow('Original', img)
cv.imshow('Kmeans', kmeans_img)
cv.imwrite('./pictures/Q4_kmeans.jpg', kmeans_img)

cv.waitKey(0)
