import numpy as np
import cv2 as cv
from mss import mss

if __name__ == "__main__":

    with mss() as sct:
        monitor = {
        "top": 0,
        "left": 350,
        "width": 1240,
        "height": 948,
        }

        # 342

        img = np.asarray(sct.grab(monitor))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dark = (gray * 0.5).astype(np.uint8)

        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        y_lower = np.array([40, 100, 50])
        y_upper = np.array([255, 255, 255])
        y_line = cv.inRange(hsv, y_lower, y_upper)

        w_lower = np.array([0, 0, 240])
        w_upper = np.array([255, 5, 255])
        w_line = cv.inRange(hsv, w_lower, w_upper)
        # weird = cv.cvtColor(dark, cv.COLOR_GRAY2HSL)

        cv.imshow("frame", w_line)
        key = cv.waitKey()

        if key == 113:
            quit()

