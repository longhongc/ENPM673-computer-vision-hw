import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Global settings
USE_ADAPTIVE = False
PLAY_VIDEO = False

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

def adaptive_hist_equalization(frame): 
    equalized_frame = frame.copy()
    h, w = frame.shape
    sub_h = int(h/2)
    sub_w = int(w/2)

    tile = np.zeros((sub_h, sub_w), dtype='uint8')
    for x in range(0, 24):
        for y in range(0, 2): 
            p1 = (x * int(sub_w/12), y * sub_h)
            p2 = (min(x * int(sub_w/12) + sub_w, w), (y + 1) * sub_h)
            tile = frame[p1[1]:p2[1], p1[0]:p2[0]].copy() 
            hist = cal_hist(tile)
            cdf = create_cdf(hist)
            equalized = hist_equalization(tile, cdf)
            equalized_frame[p1[1]:p2[1], p1[0]:p2[0]] = equalized

    return equalized_frame

    

def make_equalized(frame): 
    if len(frame.shape) == 2: 
        # gray
        hist = cal_hist(frame)
        cdf = create_cdf(hist)
        gray_equalized = hist_equalization(frame, cdf)
        equalized_hist = cal_hist(gray_equalized)

        return hist, cdf, gray_equalized

    else:
        # rgb
        colors = ['b', 'g', 'r']

        b_img, g_img, r_img = cv.split(frame)

        b_hist = cal_hist(b_img)
        g_hist = cal_hist(g_img)
        r_hist = cal_hist(r_img)

        b_cdf = create_cdf(b_hist)
        g_cdf = create_cdf(g_hist)
        r_cdf = create_cdf(r_hist)

        b_equal = hist_equalization(b_img, b_cdf)
        g_equal = hist_equalization(g_img, g_cdf)
        r_equal = hist_equalization(r_img, r_cdf)

        merged_equal = cv.merge([b_equal, g_equal, r_equal])

        return b_hist, b_cdf, merged_equal

def play_equalized_video(): 
    for i in range(0, 25):
        img_path = './adaptive_hist_data/' + str(i).rjust(10, '0') + '.png'
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if not USE_ADAPTIVE: 
            b_hist, b_cdf, rgb_equalized = make_equalized(img)
            cv.imshow('rgb_equalized', rgb_equalized)
        else: 
            adaptive_equalized = adaptive_hist_equalization(gray)
            cv.imshow('adaptive_equalized', adaptive_equalized)

        cv.waitKey(1)

def main(): 
    if not PLAY_VIDEO: 
        img = cv.imread('./adaptive_hist_data/0000000000.png')
        cv.imshow('original', img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('gray', gray)

        if not USE_ADAPTIVE: 
            gray_hist, gray_cdf, gray_equalized = make_equalized(gray)
            cv.imshow('gray_equalized', gray_equalized)

            equalized_hist = cal_hist(gray_equalized)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_title('Before Histogram Equalization')
            ax2.set_title('After Histogram Equalization')
            ax1.plot(gray_hist)
            ax2.plot(equalized_hist)
            ax1.set_xlim([0, 256])
            ax2.set_xlim([0, 256])
            plt.show()

            b_hist, b_cdf, rgb_equalized = make_equalized(img)
            cv.imshow('rgb_equalized', rgb_equalized)
        else: 
            equalized = adaptive_hist_equalization(gray)
            cv.imshow('adaptive_equalized', equalized)

            gray_hist, gray_cdf, gray_equalized = make_equalized(gray)
            equalized_hist = cal_hist(equalized)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_title('Before Adaptive')
            ax2.set_title('After Adaptive')
            ax1.plot(gray_hist)
            ax2.plot(equalized_hist)
            ax1.set_xlim([0, 256])
            ax2.set_xlim([0, 256])
            plt.show()


        cv.waitKey(0)
    else:
        play_equalized_video()


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--adaptive", help="Apply adaptive historgram equalization", action="store_true")
    parser.add_argument("-v", "--video", help="Play the whole video after equalization, defualt: show single frame", action="store_true")
    args = parser.parse_args()

    USE_ADAPTIVE = args.adaptive
    PLAY_VIDEO = args.video

    main()
