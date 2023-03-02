import numpy as np
import cv2 as cv

image_points = np.array([[757, 213],
                         [758, 415], 
                         [758, 686],
                         [759, 966],
                         [1190, 172],
                         [329, 1041],
                         [1204, 850],
                         [340, 159]])
homo_col = np.ones((8, 1))
h_image = np.hstack((image_points, homo_col))

world_points = np.array([[0, 0, 0],
                         [0, 3, 0],
                         [0, 7, 0],
                         [0, 11, 0],
                         [7, 1, 0],
                         [0, 11, 7],
                         [7, 9, 0],
                         [0, 1, 7]])
h_world = np.hstack((world_points, homo_col))

def create_m_row(image_point, world_point):
    u, v, w = image_point
    a, b, c, d = world_point 
    return np.array([[  a,  b,  c,  d,  0,  0,  0,  0, -u*a, -u*b, -u*c, -u*d],
                     [  0,  0,  0,  0,  a,  b,  c,  d, -v*a, -v*b, -v*c, -v*d]])

# Create A matrix, Am=0
rows = []
for i in range(0, 8): 
    rows.append(create_m_row(h_image[i], h_world[i]))

A = np.vstack(rows)
u, w, vh = np.linalg.svd(A, full_matrices=True)
P = vh[-1].reshape((3,4))

# Calculate camera position
u, w, vh = np.linalg.svd(P, full_matrices=True)
C = vh[-1]
C = C / C[-1]
C = np.array([C]).T[:3]
print("C matrix: ")
print(C)

# Calculate KR
KR = P.dot(np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1], 
                     [0, 0, 0]]))

M = KR

# Calculate K and R
c_x = -M[2, 2] / np.sqrt(M[2, 1]**2 + M[2,2]**2) 
s_x =  M[2, 1] / np.sqrt(M[2, 1]**2 + M[2,2]**2) 

R_x = np.array([[1,   0,    0],
                [0, c_x, -s_x],
                [0, s_x,  c_x]])

MR_x = M.dot(R_x)
c_y =  MR_x[2, 2] / np.sqrt(MR_x[2, 0]**2 + MR_x[2,2]**2) 
s_y =  MR_x[2, 0] / np.sqrt(MR_x[2, 0]**2 + MR_x[2,2]**2) 

R_y = np.array([[ c_y,   0,    s_y],
                [   0,   1,      0],
                [-s_y,   0,    c_y]])

MR_xR_y= MR_x.dot(R_y)
c_z =   MR_xR_y[1, 1] / np.sqrt(MR_xR_y[1, 0]**2 + MR_xR_y[1,1]**2) 
s_z =  -MR_xR_y[1, 0] / np.sqrt(MR_xR_y[1, 0]**2 + MR_xR_y[1,1]**2) 

R_z = np.array([[ c_z, -s_z,  0],
                [ s_z,  c_z,  0],
                [   0,    0,  1]])

K = MR_xR_y.dot(R_z)
# Recover the K with a scalar to make the last element 1
K = K * (1/K[2,2])
print("K matrix: ")
print(K)
R = R_z.T.dot(R_y.T.dot(R_x.T))
print("R matrix: ")
print(R)

I = np.identity(3)
P = K.dot(R.dot(np.hstack((I, -C)))) 
print("P matrix")
print(P)

print("---Testing---")
print("world points: ", world_points[7])
print("groud truth image points: ", image_points[7])
cal_points = P.dot(h_world[7])
cal_points = cal_points / cal_points[-1]
print("Calculated by P: ", cal_points[:2])
print("---")
print("world points: ", world_points[5])
print("groud truth image points: ", image_points[5])
cal_points = P.dot(h_world[5])
cal_points = cal_points / cal_points[-1]
print("Calculated by P: ", cal_points[:2])
