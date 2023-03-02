import cv2
import numpy as np
import matplotlib.pyplot as plt

def video_to_frame(video_path):
    """
        Extract the video into images for later use

        input: video path
    """
    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened() == False):
        print('Error opening video stream or file')

    count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:

            cv2.imwrite('./frame/frame{}.jpg'.format(count), frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break 
        else:
            break
        count += 1
        print(count)

    cap.release()

def draw_spectrum(f_shift):
    """
        Draw the frequency spectrum
    """

    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    cv2.imwrite('spectrum.jpg', magnitude_spectrum)

def high_pass_filter(img):
    """
        Apply high pass filter to an image

        input: image
        return: filterd image
    """
    # fft transform
    f_img = np.fft.fft2(img)

    # shift the zero frequency to the center
    f_shift = np.fft.fftshift(f_img)

    rows, cols = img.shape
    row_c = int(rows/2)
    col_c = int(cols/2)
    mask_l = 80
    # mask the middle part of the frequency image
    for i in range(-mask_l, mask_l+1):
        for j in range(-mask_l, mask_l+1):
            if pow(i,2)+pow(j,2) <= pow(mask_l,2):
                f_shift[row_c+i, col_c+j] = 0.01


    # shift back 
    f_inv_shift = np.fft.ifftshift(f_shift)

    # inverse fft
    img_back = np.fft.ifft2(f_inv_shift)
    return np.uint8(np.abs(img_back))

if __name__ == '__main__':
    video_to_frame("1tagvideo.mp4")
    img = cv2.imread('frame/frame200.jpg', cv2.IMREAD_GRAYSCALE)
    img_filtered = high_pass_filter(img)
    cv2.imshow('fft', img_filtered)
    cv2.waitKey(0)
