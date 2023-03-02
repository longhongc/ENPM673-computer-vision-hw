import cv2 as cv
import numpy as np

# Global settings
SAVE = True

def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def pipeline(frame):
    frame_copy = frame.copy()

    # canvas for detected lane
    detected_lane = np.zeros((frame.shape), dtype='uint8')

    # find white dash line
    gray = cv.cvtColor(frame_copy, cv.COLOR_BGR2GRAY)
    _, masked_white = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    
    h, w = masked_white.shape
    masked_white[:, int(0.9 * w):] = 0
    masked_white[:, :int(0.75 * w)] = 0

    white_lane = np.where(masked_white == 255)

    # find yello line
    hsv = cv.cvtColor(frame_copy, cv.COLOR_BGR2HSV)
    lower = np.array([20, 120, 120], dtype="uint8")
    upper = np.array([30, 255, 255], dtype="uint8")
    masked_yello = cv.inRange(hsv, lower, upper)
    yello_lane = np.where(masked_yello == 255)

    # fit curve by points
    def fit_curve(points): 
        y, x = points
        fit_points = []
        if len(x) != 0: 
            curve_param = np.polyfit(np.array(y), np.array(x), 2)
            curve = np.poly1d(curve_param)

            for h in range(0, frame.shape[0]): 
                w = np.int(curve(h))
                if w >= frame.shape[1] or w < 0: 
                    continue
                fit_points.append((w, h))

        return curve_param, fit_points


    def draw_points(my_points, my_img, color): 
        for i, point in enumerate(my_points): 
            w, h = point
            # draw average mid lane
            if color == (255, 255, 255) and i%10==0:
                continue
            my_img[h][w] = color

    def cal_curvature(curve_param, point): 
        a, b, c = curve_param
        first_d = 2 * a * point + b * point
        second_d = 2 * a

        R = np.sqrt((1 + first_d**2)**3) / np.abs(second_d)

        return R
        

    # white lane
    white_lane_curve_param, white_lane_points = fit_curve(white_lane) 
    white_curvature = cal_curvature(white_lane_curve_param, int(frame_copy.shape[0] / 2))
    white_points = np.array(white_lane_points)
    draw_points(white_points, detected_lane, (0, 255, 0))

    # yello lane
    yello_lane_curve_param, yello_lane_points = fit_curve(yello_lane) 
    yello_curvature = cal_curvature(yello_lane_curve_param, int(frame_copy.shape[0] / 2))
    yello_points = np.array(yello_lane_points)
    draw_points(yello_points, detected_lane, (0, 0, 255))

    # average lane
    mid_lane = np.where(cv.cvtColor(detected_lane, cv.COLOR_BGR2GRAY) != 0)
    mid_lane_curve_param, mid_lane_points = fit_curve(mid_lane) 
    average_curvature = cal_curvature(mid_lane_curve_param, int(frame_copy.shape[0] / 2))
    average_points = np.array(mid_lane_points)
    draw_points(average_points, detected_lane, (255, 255, 255))


    lane_img = cv.cvtColor(cv.bitwise_or(masked_white, masked_yello), cv.COLOR_GRAY2BGR)

    points = np.vstack((yello_points, np.flip(white_points, 0)))
    lane_coverage = detected_lane.copy()
    cv.fillPoly(lane_coverage, [points], (102, 255, 178))

    result  = {"lane_image": lane_img,
               "detected_lane": detected_lane, 
               "lane_coverage": lane_coverage, 
               "white_curve": white_points,
               "white_lane_curvature": white_curvature,
               "yello_curve": yello_points,
               "yello_lane_curvature": yello_curvature,
               "average_curvature": average_curvature}

    return result 

# draw area of interest
def draw_red(frame): 
    frame_copy = frame.copy()
    p1 = (420, 349)
    p2 = (571, 349)
    p3 = (212, 446)
    p4 = (740, 446)

    color = (0, 0, 255)
    points = np.array([p1, p2, p4, p3])
    cv.line(frame_copy, p1, p2, (51, 51, 255), 3)
    cv.line(frame_copy, p2, p4, (51, 51, 255), 3)
    cv.line(frame_copy, p3, p4, (51, 51, 255), 3)
    cv.line(frame_copy, p1, p3, (51, 51, 255), 3)

    cv.circle(frame_copy, p1, 10, color, -1)
    cv.circle(frame_copy, p2, 10, color, -1)
    cv.circle(frame_copy, p3, 10, color, -1)
    cv.circle(frame_copy, p4, 10, color, -1)

    return frame_copy

# warp the image to topview
def warp_to_top_view(frame): 
    frame_copy = frame.copy()
    p1 = (420, 349)
    p2 = (571, 349)
    p3 = (212, 446)
    p4 = (740, 446)

    h, w = frame.shape[:2]
    pts1 = np.float32([p1, p2, p3, p4])
    pts2 = np.float32([[int(w/10), int(h/10)], [int(w*9/10), int(h/10)], [int(w/10), int(h*9/10)], [int(w*9/10), int(h*9/10)]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(frame_copy, M, (w, h))

    return M, dst

def play_video(): 
    cap = cv.VideoCapture('./challenge.mp4')

    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        return 

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            # rescale the image to be smaller
            smaller_frame = rescale(frame)
            h, w = smaller_frame.shape[:2]
            canvas = np.zeros((int(h * 1.25), w, 3) , dtype='uint8')
            canvas[:h, :w] = smaller_frame

            # find the topview
            transform, topview = warp_to_top_view(smaller_frame)
            smaller_topview = rescale(topview, scale = 0.25)
            small_h, small_w = smaller_topview.shape[:2]
            canvas[h:h+small_h, :small_w] = smaller_topview
            cv.rectangle(canvas, (0, h), (small_w, h+small_h-1), (255, 255, 204), thickness=1)

            result = pipeline(smaller_topview)
            # lane image
            canvas[h:h+small_h, small_w:int(small_w * 2)] = result["lane_image"] 
            cv.rectangle(canvas, (small_w, h), (int(small_w * 2), h+small_h-1), (255, 255, 204), thickness=1)

            # detected_lane
            canvas[h:h+small_h, int(small_w * 2):int(small_w * 3)] = result["detected_lane"] 
            cv.rectangle(canvas, (int(small_w * 2), h), (int(small_w * 3), h+small_h-1), (255, 255, 204), thickness=1)

            # project curve back to original image
            result_big = pipeline(topview)
            recovered_lane_coverage = cv.warpPerspective(result_big["lane_coverage"], np.linalg.inv(transform), (w, h))
            coverage_region = np.where(recovered_lane_coverage != (0, 0, 0)) 
            projected_final_image = cv.addWeighted(smaller_frame, 0.8, recovered_lane_coverage, 0.2, 0.0)
            canvas[:h, :w] = projected_final_image

            # curvature text
            text1_center = (int(small_w * 3) + 9, int(h + small_h/6))
            text2_center = (int(small_w * 3) + 9, int(h + small_h*2/6))
            text3_center = (int(small_w * 3) + 9, int(h + small_h*3/6))
            text4_center = (int(small_w * 3) + 9, int(h + small_h*4/6))
            text1 = "White Lane Curvature: {}".format(int(result["white_lane_curvature"]))    
            text2 = "Yello Lane Curvature: {}".format(int(result["yello_lane_curvature"]))  
            text3 = "Average Curvature: {}".format(int(result["average_curvature"]))
            text4 = "Turning right" if int(result["average_curvature"]) > 0 else "Turning left"
            cv.putText(canvas, text1, text1_center, cv.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0), thickness=1)
            cv.putText(canvas, text2, text2_center, cv.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 255), thickness=1)
            cv.putText(canvas, text3, text3_center, cv.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255), thickness=1)
            cv.putText(canvas, text4, text4_center, cv.FONT_HERSHEY_TRIPLEX, 0.4, (51, 153, 255), thickness=1)

            cv.imshow('predict_turn', canvas)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            break

        count += 1

    cap.release()

def main(): 
    play_video()

if __name__ == '__main__':
   main()
