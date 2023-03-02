import cv2
import numpy as np
import matplotlib.pyplot as plt

from problem1a import high_pass_filter

def find_corner_shitomasi(img):
    """
        This function is for finding corners in a image

        input: a gray scale image
        return: detected corner using shi-tomasi methond
    """
    img_canny = cv2.Canny(img, 400, 800)
    gray = np.uint8(img_canny)
    corners = cv2.goodFeaturesToTrack(gray, 28, 0.05, 15)
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
        # if distance smaller than margin, the corner is selected
        if (abs(x - x_center) < margin*x_std) and (abs(y - y_center) < margin*y_std):
            selected.append(i)

    return corners[selected]

# ax+by+d=0
def create_line(p1, p2):
    """
        This function creates line from two positions
         
        input: point1, point2
        return: a, b, d parameter in line equation ax+by+d=0
    """
    if p1 == p2:
        return []

    tangent_vector = (p2[0]-p1[0], p2[1]-p1[1])
    if (tangent_vector[0]==0):
        normal_vector = (1,0)
    elif (tangent_vector[1]==0): 
        normal_vector = (0,1)
    else:
        normal_vector = (1/(p2[0]-p1[0]), -1/(p2[1]-p1[1]))
    a, b = normal_vector
    norm = np.sqrt(pow(a, 2) + pow(b, 2))
    a, b = a / norm, b / norm 
    d = -(a * p1[0] + b * p1[1])
    return a, b, d

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

def remove_outer_edge(corners, vertics):
    """
        This function uses four vertics of a rectangle to form a boundary to preserve the corners inside. 

        input: corners, four vertics
        return: corners inside four vertics
    """
    line1 = create_line(tuple(vertics[0]), tuple(vertics[1]))
    line2 = create_line(tuple(vertics[1]), tuple(vertics[2]))
    line3 = create_line(tuple(vertics[2]), tuple(vertics[3]))
    line4 = create_line(tuple(vertics[3]), tuple(vertics[0]))

    if len(line1)==0 or len(line2)==0  or len(line3)==0 or len(line4)==0:
        return []

    selected = []
    for row in range(0, corners.shape[0]): 
        x, y = corners[row]
        if ((line1[0] * x + line1[1] * y + line1[2]) >   8 and \
            (line2[0] * x + line2[1] * y + line2[2]) <  -8 and \
            (line3[0] * x + line3[1] * y + line3[2]) >   8 and \
            (line4[0] * x + line4[1] * y + line4[2]) <  -8)  :  
            selected.append(row)

    return corners[selected]

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

def homography_transform(vertics, corner, img):
    """
        Calculate the homography transformation of points and image

        input: vertics, corners, image
        return: transformed vertics, transformed corners, transformed image
    """
    dim = 0
    for i in range(-1, 3):
        dim += np.linalg.norm(vertics[i+1] - vertics[i])
    dim = int(dim/4)+1

    # set up the square destination image
    xp = [dim, 0, 0, dim]
    yp = [dim, dim, 0, 0]

    # source image
    x = vertics[:, 0]
    y = vertics[:, 1]

    H = homography(x, y, xp, yp)
    homo_vertics = np.hstack((vertics, np.ones((vertics.shape[0],1))))
    vertics_trans = H.dot(homo_vertics.T)
    scalar = dim / vertics_trans[0, 0]
    vertics_trans *= scalar

    homo_corners = np.hstack((corners, np.ones((corners.shape[0],1))))
    corners_trans = H.dot(homo_corners.T)
    corners_trans *= scalar

    rows, cols = img.shape
    row_min = min(vertics[:, 1])
    row_max = max(vertics[:, 1])
    col_min = min(vertics[:, 0])
    col_max = max(vertics[:, 0])

    # transform the nearby pixel of the tag by homography transformation
    img_trans = np.zeros((int(3*dim), int(3*dim)))
    for i in range(0, int(1.5*dim+1)):
        for j in range(0, int(1.5*dim+1)):
            x = col_min + j
            y = row_min + i
            vec = np.array([[x, y, 1]]).T
            vec_trans = np.int0(H.dot(vec) * scalar)
            if(np.any(vec_trans < 0)):
                continue
            xp, yp, _ = vec_trans.T.tolist()[0]
            img_trans[yp, xp] = img[y, x]

    ret, thresh = cv2.threshold(img_trans, 127, 255, cv2.THRESH_BINARY)

    return np.int0(vertics_trans.T[:,:2]), np.int0(corners_trans.T[:,:2]), thresh

def decode_image(vertics, corners, img):
    """
        Decode the binary representation within the image

        input: vertics of the tag, corners, transformed image
        return: decoded tag_id
    """
    dim = vertics[0][0]
    tag = img[0:dim+1, 0:dim+1]
    tile_l = int(dim/8)
    new_ax = np.linspace(0, dim+1, 9)
    new_ax = np.int0(new_ax)

    # transformed the image into 8x8 binary image
    decoded_img = np.zeros((8, 8))
    decoded_img = decoded_img.flatten()
    # set the value of the binary image by finding the median of a window in the original image
    for i in range(0, 8):
        for j in range(0, 8): 
            code = np.median(tag[new_ax[i]: new_ax[i+1], new_ax[j]: new_ax[j+1]])
            code = 1 if code>0 else 0
            decoded_img[i*8+j] = code

    decoded_img = decoded_img.reshape((8,8))
    bin_codes = [(decoded_img[2,2], decoded_img[3,3]), 
                 (decoded_img[2,5], decoded_img[3,4]),
                 (decoded_img[5,5], decoded_img[4,4]),
                 (decoded_img[5,2], decoded_img[4,3])]

    # rotate
    find_orient = False
    for i in range(0, 4):
        if bin_codes[i][0] > 0:
            find_orient = True
            break; 

    if not find_orient:
        return -1 

    # rotate the inner tile according to the outer tile value
    while(bin_codes[2][0] != 1):
        bin_codes = bin_codes[1:] + bin_codes[:1]

    # calculate binary representation
    tag_id=0
    for i in range(0, 4):
        tag_id += bin_codes[i][1] * pow(2, i)

    return tag_id

if __name__ == '__main__': 
    count = 0
    # read image frame
    # Didn't read full video because this method is not fast
    for i in range(100, 200): 
        img_path = 'frame/frame{}.jpg'.format(i)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = gray.copy()
        #ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        # apply threshold
        img[np.where(img < 130)] = 0

        # find corners
        corners = find_corner_shitomasi(img)

        if len(corners) < 4:
            print("frame{} detection fails".format(i))
            count += 1
            continue

        # remove outlier corners
        corners = find_middle_corners(corners, "median", 2.8)

        # find the outer vertics 
        vertics = find_outer_vertics(corners)

        # find the inner corners 
        corners = remove_outer_edge(corners, vertics)

        if len(corners) < 4:
            print("frame{} detection fails".format(i))
            count += 1
            continue

        # find the outer vertics of the tag
        vertics = find_outer_vertics(corners)
        try:
            # homography transformation
            vertics_trans, corner_trans, img_trans = homography_transform(vertics, corners, img)
            # decode 
            tag_id = decode_image(vertics_trans, corner_trans, img_trans)
            if tag_id > 15 or tag_id < 0:
                count += 1
                continue
            print("frame{}: {}".format(i, tag_id))

        except:
            print("frame{} detection fails".format(i))
            count += 1
            continue

        # add corner as red spot to the image
        img_corners = print_corners2(img, vertics)
        cv2.imshow('ar_tag', img_corners)
        cv2.waitKey(10)

    print("detection fails: ", count)

    cv2.waitKey(0)

