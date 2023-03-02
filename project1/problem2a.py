import cv2
import numpy as np

def find_corner_shitomasi(img, points=28):
    """
        This function is for finding corners in a image

        input: a gray scale image
        return: detected corner using shi-tomasi methond
    """
    img_canny = cv2.Canny(img, 400, 800)
    gray = np.uint8(img_canny)
    corners = cv2.goodFeaturesToTrack(gray, points, 0.01, 20)
    if(type(corners) == type(None)):
        return []
    corners = np.int0(corners)
    corners = np.reshape(corners, (corners.shape[0], 2))

    return corners

def print_corners(img, corners): 
    """
        This function add corners as red spot on to an image

        input: a gray scale image, position of corners
        return: a rgb image
    """
    img_canny = cv2.Canny(img, 400, 800)
    gray = np.uint8(img_canny)
    backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    for corner in corners:
        x, y = corner
        cv2.circle(backtorgb, (x,y), 5, [0, 0, 255], -1)

    return backtorgb

def print_corners2(img, corners): 
    """
        This function add corners as red spot on to an image

        input: a rgb image, position of corners
        return: a rgb image
    """
    gray = np.uint8(img)
    img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    for corner in corners:
        x, y = corner
        cv2.circle(img, (x,y), 5, [0, 0, 255], -1)

    return img

def find_middle_corners(corners, center="median", margin=1):
    """
        This function find corners that are closer to the center of all corners
        The purpose of this funciton is to reject outlier corners

        input: position of corners, method for finding center (mean or median), how much standard deviation should be selected
        return: selected corners
    """

    corners_x = corners[:,0]
    corners_y = corners[:,1]
    x_std = np.std(corners_x)
    y_std = np.std(corners_y)

    if center == "median": 
        x_center = np.median(corners_x)
        y_center = np.median(corners_y)
    elif center == "mean":
        x_center = np.mean(corners_x)
        y_center = np.mean(corners_y)

    selected = []
    rows, cols = corners.shape
    for i in range(0, rows): 
        x, y = corners[i]
        if (abs(x - x_center) < margin*x_std) and (abs(y - y_center) < margin*y_std):
            selected.append(i)

    return corners[selected]

def find_outer_vertics(corners): 
    """
        This function finds the outer corners in all the corners.
        Outer corners means top-most, left-most, bottom-most, and right-most corners. 

        input: corners
        return: outer corners
    """

    corners_x = corners[:,0]
    corners_y = corners[:,1]

    max_x_i = np.where(corners_x==corners_x.max())
    min_x_i = np.where(corners_x==corners_x.min())

    max_y_i = np.where(corners_y==corners_y.max())
    min_y_i = np.where(corners_y==corners_y.min())

    outer_vertics = np.vstack((corners[max_x_i][0], corners[max_y_i][-1], corners[min_x_i][-1], corners[min_y_i][0]))
    return outer_vertics

def homography(x, y, xp, yp):
    """
        Calculate homography through two vectors

        input: source vector x y, destination vector xp  yp
        return: the homogrphy matrix 
    """

    A = np.array([[x[0], y[0], 1, 0, 0, 0, -x[0]*xp[0], -y[0]*xp[0], -xp[0]],
                  [0, 0, 0, x[0], y[0], 1, -x[0]*yp[0], -y[0]*yp[0], -yp[0]],
                  [x[1], y[1], 1, 0, 0, 0, -x[1]*xp[1], -y[1]*xp[1], -xp[1]],
                  [0, 0, 0, x[1], y[1], 1, -x[1]*yp[1], -y[1]*yp[1], -yp[1]],
                  [x[2], y[2], 1, 0, 0, 0, -x[2]*xp[2], -y[2]*xp[2], -xp[2]],
                  [0, 0, 0, x[2], y[2], 1, -x[2]*yp[2], -y[2]*yp[2], -yp[2]],
                  [x[3], y[3], 1, 0, 0, 0, -x[3]*xp[3], -y[3]*xp[3], -xp[3]],
                  [0, 0, 0, x[3], y[3], 1, -x[3]*yp[3], -y[3]*yp[3], -yp[3]] ])

    u, w, vh = np.linalg.svd(A, full_matrices=True)
    return vh[-1].reshape((3,3))

def wrap_image(vertics, img1, img2):
    """
        Superimposing image(img2) onto tag(img1)

        input: vertics of the tag, the tag(img1), the target image(img2)
    """
    dim1 = 0
    for i in range(-1, 3):
        dim1 += np.linalg.norm(vertics[i+1] - vertics[i])
    dim1 = int(dim1/4)+1

    dim2_y, dim2_x, _ = img2.shape
    xp = [dim2_x, 0, 0, dim2_x]
    yp = [dim2_y, dim2_y, 0, 0]
    x = vertics[:, 0]
    y = vertics[:, 1]

    H = homography(x, y, xp, yp)
    homo_vertics = np.hstack((vertics, np.ones((vertics.shape[0],1))))
    vertics_trans = H.dot(homo_vertics.T)
    scalar = dim2_x / vertics_trans[0, 0]

    # changing the value in the tag image by the value in target image
    for i in range(0, dim2_y):
        for j in range(0, dim2_x):
            homo_img2_point = np.array([[j], [i], [1]])
            homo_img1_point = np.linalg.inv(H).dot(homo_img2_point)
            homo_img1_point = homo_img1_point / scalar
            x, y, _ = np.int0(homo_img1_point).T.tolist()[0]
            img1[y, x] = img2[i, j] 

    return img1

if __name__ == '__main__': 
    testudo = cv2.imread("testudo.png")
    count = 0
    for i in range(100, 101): 
        img_path = 'frame/frame{}.jpg'.format(i)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imread(img_path)
        img = gray.copy()

        # apply threshold
        #ret, thresh = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
        img[np.where(img < 130)] = 0

        # find corners
        corners = find_corner_shitomasi(img, 18)

        if len(corners) < 4:
            print("frame{} detection fails".format(i))
            count += 1
            continue

        # remove outlier corners
        corners = find_middle_corners(corners, "mean", 1.8)
        if len(corners) < 4:
            print("frame{} detection fails".format(i))
            count += 1
            continue

        # remove outer corners
        corners = find_middle_corners(corners, "mean", 2.2)
        if len(corners) < 4:
            print("frame{} detection fails".format(i))
            count += 1
            continue

        # find the inner tag corners 
        vertics = find_outer_vertics(corners)
        try:
            # wrap the image
            img_trans = wrap_image(vertics, img_color, testudo)
        except:
            print("frame{} detection fails".format(i))
            count += 1
            continue

        cv2.imshow('ar_tag', img_trans)
        cv2.waitKey(10)

    print("detection fails: ", count)
    cv2.waitKey(0)
