import cv2 as cv
import numpy as np

MIN_MATCH_COUNT = 10

img1 = cv.imread('./pictures/Q2imageA.png')
img2 = cv.imread('./pictures/Q2imageB.png')

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT feature detection
sift = cv.SIFT_create()

key1, des1 = sift.detectAndCompute(gray1, None)
key2, des2 = sift.detectAndCompute(gray2, None)

bf = cv.BFMatcher()

# Find two matches for every descriptor
matches = bf.knnMatch(des2, des1, k=2)

good_matches = []
for m in matches:
    if m[0].distance < 0.5 * m[1].distance:
        good_matches.append([m[0]])

if len(good_matches) > MIN_MATCH_COUNT: 
    match_img = cv.drawMatchesKnn(img1, key1, img2, key2, good_matches, None, flags=2)
    cv.imshow('match points', match_img)
    cv.imwrite('./pictures/Q2_match_points.jpg', match_img)

    # reshape good matches for homography
    src = np.float32([key2[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([key1[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src, dst, cv.RANSAC, 4.0)

    stitched = cv.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0] + img2.shape[0]), cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0))

    stitched[0:img1.shape[0], 0:img1.shape[1]] = img1

    cv.imshow('stitched', stitched)
    cv.imwrite('./pictures/Q2_stitched.jpg', stitched)
else:
    print("Not enough matching point")

cv.waitKey(0)

