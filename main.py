import numpy as np
import matplotlib.pyplot as plt
import time
import cv2 as cv
from mss import mss

def get_img():
    with mss() as sct:
        img = np.asarray(sct.grab(monitor))
        return img


def birds_eye(img):
    t_matrix = cv.getPerspectiveTransform()

if __name__ == "__main__":
    monitor = {
    "top": 30,
    "left": 350,
    "width": 1240,
    "height": 918,
    }
    plt.ion()
    plt.show()

    while True:
        img = get_img()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dark = (gray * 0.5).astype(np.uint8)

        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        y_lower = np.array([100, 140, 100])
        y_upper = np.array([255, 255, 255])
        y_line = cv.inRange(hsv, y_lower, y_upper)

        w_lower = np.array([18, 0, 200])
        w_upper = np.array([255, 10, 255])
        w_line = cv.inRange(hsv, w_lower, w_upper)

        # combine two masks
        combined = y_line | w_line

        blur = cv.blur(combined, (5, 5))

        edges = cv.Canny(combined, 100, 200)

        trapezoid = np.array([
        [
        # top left
        (620-120, 520), # 494
        # bottom left
        (40, 918-120),
        # bottom right
        (1240-40, 918-120),
        # top right
        (620+120, 520)]
        ])

        mask = np.zeros_like(edges)

        mask = cv.fillPoly(mask, trapezoid, 255)
        mask = cv.bitwise_and(edges, mask)

        padding = monitor["width"] * 0.25
        desired_points = np.array([
        [0, 0],
        [0, monitor["height"]],
        [monitor["width"], monitor["height"]],
        [monitor["width"], 0],
        ], np.float32)

        t_matrix = cv.getPerspectiveTransform(np.float32(trapezoid[0]), desired_points)
        warped_frame = cv.warpPerspective(mask, t_matrix, (monitor["width"], monitor["height"]), flags=(cv.INTER_LINEAR))

        historigram = np.sum(warped_frame[int(warped_frame.shape[0]/2):,:], axis=0)
        plt.clf()
        plt.plot(historigram)
        plt.draw()
        plt.pause(0.001)

        margin = int(monitor["width"] * (1/12))
        window_height = int(warped_frame.shape[0] / 10)

        midpoint = int(historigram.shape[0] / 2)
        left_x = np.argmax(historigram[:midpoint])
        right_x = np.argmax(historigram[midpoint:]) + midpoint

        non_zero = warped_frame.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        cv.imshow("frame", warped_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

