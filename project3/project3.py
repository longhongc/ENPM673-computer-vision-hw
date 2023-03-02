import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
from tqdm import tqdm

MIN_MATCH_COUNT = 10

def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def feature_matching(img1, img2, picture): 
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
        if m[0].distance < 0.7 * m[1].distance:
            good_matches.append([m[0]])

    if len(good_matches) > MIN_MATCH_COUNT: 
        match_img = cv.drawMatchesKnn(img1, key1, img2, key2, good_matches, None, flags=2)
        cv.imshow("match points", match_img)
        cv.imwrite("./pictures/" + picture + "_match_points.jpg", match_img)

        # reshape good matches for homography
        src = np.float32([key2[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst = np.float32([key1[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 2)

        return dst, src

def cal_fundamental_matrix(img1_points, img2_points):
    # Create A matrix, Am=0
    rows = []
    for i in range(0, 8): 
        x1, y1 = img1_points[i]
        x2, y2 = img2_points[i]
        rows.append(np.array([[x1*x2, y1*x2, x2, y2*x1, y1*y2, y2, x1, y1, 1]]))

    A = np.vstack(rows)
    u, w, vh = np.linalg.svd(A, full_matrices=True)
    f = vh[-1].reshape((3,3))
    f_u, f_w, f_vh = np.linalg.svd(f, full_matrices=True)

    # Make F rank 2
    f_w[-1] = 0
    rectified_f = f_u.dot(np.diag(f_w).dot(f_vh))

    return rectified_f

# e is the probability of outlier data, outlier data/total data
# p is the probability to have good model in RANSAC
# s is the number of sampled points
def RANSAC(points_1, points_2, e=0.4, p=0.95, s=8):
    N = int(math.log(1-p)/math.log(1-pow((1-e),s)))
    print("Iterations: ", N)
    points_count = points_1.shape[0]
    model_eval = {}
    for _ in range(0, N):
        try:
            sampled_index = np.random.choice(points_count, s, replace=False)
            F_matrix = cal_fundamental_matrix(points_1[sampled_index], points_2[sampled_index])

            total_error = 0
            points_error = []
            for i in range(points_count): 
                p1 = np.append(points_1[i], 1).reshape(3,1)
                p2 = np.append(points_2[i], 1).reshape(1,3)
                error = p2.dot(F_matrix).dot(p1).reshape(1)[0]
                points_error.append(abs(error))
                total_error += abs(error)

            average_error = total_error / points_count

            inlier_index = np.where(np.array(points_error) < average_error * 0.05)[0]
            inlier_count = inlier_index.shape[-1]
            inlier_error = sum(np.array(points_error)[inlier_index]) / inlier_count
            model_eval[inlier_count] = (F_matrix, inlier_index)
        except:
            pass

    print("Max inlier: ", max(model_eval.keys()))
    return model_eval[max(model_eval.keys())]

def cal_essensial_matrix(F, K): 
    E = K.T.dot(F.dot(K))
    u, w, vh = np.linalg.svd(E, full_matrices=True)

    # Make E rank 2
    w[-1] = 0
    rectified_E = u.dot(np.diag(w).dot(vh))

    return rectified_E

def cal_camera_pose_from_E(E, K, good_matches): 
    points_1, points_2 = good_matches
    homo_points_1 = np.hstack((points_1, np.ones((points_1.shape[0], 1))))
    homo_points_2 = np.hstack((points_2, np.ones((points_2.shape[0], 1))))

    # algebraic method (Hartley & Zisserman)
    u, _, vh = np.linalg.svd(E, full_matrices=True)
    w = np.array([1,1,0])
    Z1 = np.array([[ 0,  1,  0],
                   [-1,  0,  0],
                   [ 0,  0,  0]])
    W1 = np.array([[ 0, -1,  0],
                   [ 1,  0,  0],
                   [ 0,  0,  1]])
    S1 = u.dot(Z1.dot(u.T))
    R1 = u.dot(W1.dot(vh))
    t1 = np.array([[S1[2,1], S1[0,2], S1[1,0]]]).T
    M1 = np.hstack((R1, t1))

    Z2 = -Z1.T 
    W2 = W1
    S2 = u.dot(Z2.dot(u.T))
    R2 = u.dot(W2.dot(vh))
    t2 = np.array([[S2[2,1], S2[0,2], S2[1,0]]]).T
    M2 = np.hstack((R2, t2))

    Z3 = -Z1
    W3 = W1.T
    S3 = u.dot(Z3.dot(u.T))
    R3 = u.dot(W3.dot(vh))
    t3 = np.array([[S3[2,1], S3[0,2], S3[1,0]]]).T
    M3 = np.hstack((R3, t3))

    Z4 = Z1.T
    W4 = W1.T
    S4 = u.dot(Z4.dot(u.T))
    R4 = u.dot(W4.dot(vh))
    t4 = np.array([[S4[2,1], S4[0,2], S4[1,0]]]).T
    M4 = np.hstack((R4, t4))

    M_valid = [True] * 4
    for i in range(points_1.shape[0]):
        p2 = homo_points_2[i]
        p2_w_in_2 = np.linalg.inv(K).dot(np.array([p2]).T)
        p2_w1 = M1.dot(np.vstack((p2_w_in_2, np.array([1]))))
        p2_w2 = M2.dot(np.vstack((p2_w_in_2, np.array([1]))))
        p2_w3 = M3.dot(np.vstack((p2_w_in_2, np.array([1]))))
        p2_w4 = M4.dot(np.vstack((p2_w_in_2, np.array([1]))))
        if p2_w1[2, 0] < 0: 
            M_valid[0] = False
        if p2_w2[2, 0] < 0: 
            M_valid[1] = False
        if p2_w3[2, 0] < 0: 
            M_valid[2] = False
        if p2_w4[2, 0] < 0: 
            M_valid[3] = False

    Ms = [M1, M2, M3, M4]
    print(M_valid)
    for i in range(4): 
        if M_valid[i]: 
            return Ms[i]

# load intrinsic data 
def load_intrinsic(picture): 
    path = "./" + picture + "/calib.txt"
    param={}
    with open(path) as f:
        for i, line in enumerate(f): 
            name, value = line.split('=')
            if i < 2: 
                rows_str = value[1:-2].split(';')
                intrinsic_matrix = np.zeros((3,3))
                for j, row_str in enumerate(rows_str):
                   row = [float(k) for k in row_str.split()]
                   intrinsic_matrix[j] = row
                param[name] = intrinsic_matrix
            else: 
                param[name] = float(value)

    return param

def cal_epipolar_lines(img1, img2, F, points_1, points_2, rectified):
    img_epi_1 = img1.copy()
    img_epi_2 = img2.copy()

    lines1 = []
    lines2 = []
    for i in range(points_1.shape[0]):
        x1 = np.array([points_1[i][0], points_1[i][1], 1]).reshape(3,1)
        x2 = np.array([points_2[i][0], points_2[i][1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        lines2.append(line2)

        line1 = np.dot(F.T, x2)
        lines1.append(line1)

        if not rectified:
            x2_min = 0
            x2_max = img2.shape[1] - 1
            y2_min = -(line2[0]*x2_min + line2[2])/line2[1]
            y2_max = -(line2[0]*x2_max + line2[2])/line2[1]

            x1_min = 0
            x1_max = img1.shape[1] - 1
            y1_min = -(line1[0]*x1_min + line1[2])/line1[1]
            y1_max = -(line1[0]*x1_max + line1[2])/line1[1]
        else:
            # lines are horizontal if rectified
            x2_min = 0
            x2_max = img2.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]

            x1_min = 0
            x1_max = img1.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]

        cv.circle(img_epi_2, np.int32(points_2[i][:2]), 10, (0,0,255), -1)
        img_epi_2 = cv.line(img_epi_2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 255, 0), 2)


        cv.circle(img_epi_1, np.int32(points_1[i][:2]), 10, (0,0,255), -1)
        img_epi_1 = cv.line(img_epi_1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 255, 0), 2)


    concat_img = rescale(np.hstack((img_epi_1, img_epi_2)), 0.5)

    return lines1, lines2, concat_img

def Calibration(img1, img2, picture):
    points_1, points_2 = feature_matching(img1, img2, picture)

    F, index = RANSAC(points_1, points_2)
    print("Fundamental Matrix: ")
    print(F)

    param = load_intrinsic(picture)

    K = param["cam0"]

    E = cal_essensial_matrix(F, K)
    print("Essential Matrix: ")
    print(E)

    good_matches = (points_1[index], points_2[index])
    M = cal_camera_pose_from_E(E, K, good_matches)
    R = M[:,:3]
    print("Rotation matrix")
    print(R)

    T = M[:,-1:]
    print("Translation")
    print(T)

    return F, points_1[index], points_2[index]

def Rectification(img1, img2, points_1, points_2, F):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # calculate H1 and H2
    _, H1, H2 = cv.stereoRectifyUncalibrated(points_1, points_2, F, imgSize=(w1, h1))

    # warp images
    img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))

    # transform the feature points 
    points_1_rectified = cv.perspectiveTransform(points_1.reshape(-1, 1, 2), H1).reshape(-1,2)
    points_2_rectified = cv.perspectiveTransform(points_2.reshape(-1, 1, 2), H2).reshape(-1,2)

    F_rectified = np.linalg.inv(H2.T).dot(F.dot(np.linalg.inv(H1)))
    print("H for left: ")
    print(H1)
    print("H for right: ")
    print(H2)
    lines1, lines2, unrectified = cal_epipolar_lines(img1, img2, F, points_1, points_2, False)
    lines1_rectified, lines2_rectified, rectified = cal_epipolar_lines(img1_rectified, img2_rectified, F_rectified, points_1_rectified, points_2_rectified, True)

    return img1_rectified, img2_rectified, unrectified, rectified, lines1, lines2

def SSD(gray1, gray2, picture): 
    param = load_intrinsic(picture)
    ndisp = param["ndisp"] 

    disparity_map = np.zeros(gray1.shape)

    kernel_size = 5
    h, w = gray1.shape
    indices = np.arange(0, ndisp). reshape(-1,1)
    indices_j = np.clip(np.arange(0, w-kernel_size+1), 0, w-kernel_size).T

    x_min = 0
    x_max = gray1.shape[1] - 1

    for i in tqdm(np.clip(range(h), kernel_size//2, h-kernel_size//2 - 1)): 
        kernels1 = []
        kernels2 = []
        for j in np.clip(range(w), kernel_size//2, w-kernel_size//2 - 1): 
            kernel1 = gray1[i-kernel_size//2:i+kernel_size//2+1, j-kernel_size//2:j+kernel_size//2+1].reshape(1,-1)
            kernel2 = gray2[i-kernel_size//2:i+kernel_size//2+1, j-kernel_size//2:j+kernel_size//2+1].reshape(1,-1)
            kernels1.append(kernel1.reshape(-1))
            kernels2.append(kernel2.reshape(-1))

        kernels1 = np.array(kernels1).astype(float)
        kernels2 = np.array(kernels2).astype(float)
        SSD = (kernels1**2).sum(-1).reshape(-1,1) + (kernels2**2).sum(-1).reshape(1,-1) - 2*(kernels1 @ kernels2.T)
        min_indexs = SSD.argmin(-1)
        for idx, j in enumerate(np.clip(range(w), kernel_size//2, w-kernel_size//2 - 1)): 
            # min_indexs[idx] is the correspondence points on the other image
            disparity_map[i, j] =  min_indexs[idx] - j


    return disparity_map
            
def Correspondence(gray1, gray2, picture): 
    disparity_map = SSD(gray1, gray2, picture)
    max_disparity = np.max(disparity_map)
    disparity_map_scaled = np.uint8(disparity_map * 255 / max_disparity)
    heatmap = cv.applyColorMap(disparity_map_scaled, cv.COLORMAP_HOT)
    cv.imshow("disparity", rescale(disparity_map, 0.5))
    cv.imshow("heatmap", rescale(heatmap, 0.5))
    cv.imwrite("./pictures/" + picture + "_disparity.png", disparity_map)
    cv.imwrite("./pictures/" + picture + "_disparity_heatmap.png", heatmap)

    return disparity_map

def cal_hist(frame): 
    hist = [0] * 256
    for i in range(0, 256): 
        hist[i] = np.sum(frame == i)

    return np.array(hist).reshape(256, 1)

def create_cdf(hist_list):
    cdf = np.zeros(hist_list.shape)
    for i in range(0, len(hist_list)):
        if i==0:
            cdf[i] = hist_list[i]
        else:
            cdf[i] = cdf[i-1] + hist_list[i]

    return cdf

def hist_equalization(frame, cdf):
    equalized_frame = frame.copy()
    height, width = frame.shape[:2]

    total = height * width

    for i in range(0, 256): 
        equalized_frame[np.where(frame == i)] = cdf[i] * 255 / total

    return np.uint8(equalized_frame)

def make_equalized(frame): 
    # gray
    hist = cal_hist(frame)
    cdf = create_cdf(hist)
    gray_equalized = hist_equalization(frame, cdf)
    equalized_hist = cal_hist(gray_equalized)

    return gray_equalized

def ComputeDepthImage(disparity_img, picture): 
    param = load_intrinsic(picture)
    K = param["cam0"]
    f = K[0,0]
    baseline = param["baseline"] 

    depth_img = np.zeros(disparity_img.shape, dtype='float')
    # depth = f * baseline / disparity
    depth_img[disparity_img > 0] = (f * baseline) / (disparity_img[disparity_img > 0])
    # resacle from 0 ~ 255
    depth_img = ((depth_img/depth_img.max())*255).astype(np.uint8)
    equalized_depth_img = make_equalized(depth_img)
    equalized_heatmap = cv.applyColorMap(equalized_depth_img, cv.COLORMAP_HOT)
    heatmap = cv.applyColorMap(depth_img, cv.COLORMAP_HOT)
    cv.imshow("depth.png", rescale(equalized_depth_img, 0.5))
    cv.imshow("heatdepth.png", rescale(equalized_heatmap, 0.5))
    cv.imwrite("./pictures/" + picture + "_depth.png", depth_img)
    cv.imwrite("./pictures/" + picture + "_equalized_depth.png", equalized_depth_img)
    cv.imwrite("./pictures/" + picture + "_heatdepth.png", heatmap)
    cv.imwrite("./pictures/" + picture + "_equalized_heatdepth.png", equalized_heatmap)



def main(): 
    print("Choose a picture (1) curule, (2) octagon, (3) pendulum (Enter 1,2,3): ")
    picture_type = int(input())
    if picture_type == 1:
        picture = "curule"
    elif picture_type == 2:
        picture = "octagon"
    elif picture_type == 3: 
        picture = "pendulum"

    print("Start processing...")

    img1_path = "./" + picture + "/im0.png"
    img2_path = "./" + picture + "/im1.png"

    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    small_img1 = rescale(img1, 0.4)
    small_img2 = rescale(img2, 0.4)
    gray1 = cv.cvtColor(small_img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(small_img2, cv.COLOR_BGR2GRAY)

    # Calibration
    F, points_1, points_2 = Calibration(img1, img2, picture)
    
    # Rectification
    img1_rectified, img2_rectified, unrectified, rectified, lines1, lines2 = Rectification(img1, img2, points_1, points_2, F)

    gray1 = cv.cvtColor(img1_rectified, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2_rectified, cv.COLOR_BGR2GRAY)

    # Correspondence
    disparity_img = Correspondence(gray1, gray2, picture)

    # ComputeDepthImage
    ComputeDepthImage(disparity_img, picture)
     
    cv.imshow("Rectified", rectified)
    cv.imshow("Unrectified", unrectified)
    cv.imwrite("./pictures/" + picture + "_rectified.png", rectified)
    cv.imwrite("./pictures/" + picture + "_unrectified.png", unrectified)
     
    
    while True:
        cv.waitKey(1)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
