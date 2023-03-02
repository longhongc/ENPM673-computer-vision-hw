import cv2 as cv
import numpy as np

# Global settings
SAVE = True

def pipeline(frame): 
    frame_copy = frame.copy()
    black = np.zeros((frame.shape[:2]), dtype='uint8')
    gray = cv.cvtColor(frame_copy, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 215, 255, cv.THRESH_BINARY)
    canny = cv.Canny(thresh, 500, 900)
    lines = cv.HoughLinesP(canny, 1, np.pi/180, 20, minLineLength=38, maxLineGap=10)

    # classify the lines found by HoughLines 
    def line_classify(my_lines): 
        lines_catagory = []
        if my_lines is not None: 
            for line in my_lines:
                if len(lines_catagory) == 0: 
                    lines_catagory.append([line])
                    continue

                dist_to_cat = []
                for cat in lines_catagory:
                    x1, y1, x2, y2 = line[0]
                    sample_line = cat[0]
                    xs1, ys1, xs2, ys2 = sample_line[0]
                    dist = np.abs((x1-xs1) * (ys2-y1) - (y1-ys1) * (xs2-x1))
                    dist_to_cat.append(dist)
                # line belongs to other line the distiance with previous line is too big
                if min(dist_to_cat) > 10000:
                    lines_catagory.append([line])
                else: 
                    idx = np.argmin(dist_to_cat)
                    lines_catagory[idx].append(line)

        return lines_catagory

    lines_catagory = line_classify(lines)

    # fit the line with the same catagory
    def lines_fitting(my_catagory): 

        x = []
        y = []
        for line in my_catagory: 
            x1, y1, x2, y2 = line[0]
            x.append(x1)
            y.append(y1)
            x.append(x2)
            y.append(y2)

        if (max(x) - min(x)) > (max(y) - min(y)):
            line_type = 'horizontal'
        else:
            line_type = 'vertical'

        # least sqaure fitting
        Y = np.array([y]).T
        X_col0 = np.array([x]).T
        X_col1 = np.ones((len(x), 1))
        X = np.hstack((X_col0, X_col1))
        B = np.matmul(np.linalg.inv(X.T.dot(X)), X.T.dot(Y)) # B = inv(X.T*X)(X.T*Y)

        if line_type == 'horizontal': 
            x1 = min(x) 
            x2 = max(x) 
            y1 = B[0] * x1 + B[1]
            y2 = B[0] * x2 + B[1]
        else:
            y1 = min(y) 
            y2 = max(y)
            x1 = (y1 - B[1]) / B[0]
            x2 = (y2 - B[1]) / B[0] 

        return int(x1), int(y1), int(x2), int(y2)

    def draw_line(my_lines_catagory, my_img): 

        for cat in my_lines_catagory: 
            dist = 0
            for line in cat: 
                x1, y1, x2, y2 = line[0]
                dist = max(dist, np.sqrt((x2-x1)**2 + (y2-y1)**2))

            if dist > 200: 
                color = (0, 255, 0)
            else:
               color = (0, 0, 255)

            x1, y1, x2, y2 = lines_fitting(cat)
            cv.line(my_img, (x1, y1), (x2, y2), color, 3)

    draw_line(lines_catagory, frame_copy)

    transformed = frame_copy 

    return transformed

def play_video(): 
    cap = cv.VideoCapture('whiteline.mp4')

    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        return 

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            transformed = pipeline(frame)
            cv.imshow('whiteline', transformed)

            if cv.waitKey(20) & 0xFF == ord('q'):
                break

        else:
            break

        count += 1

    cap.release()

def main(): 
    play_video()

if __name__ == '__main__':
    main()
