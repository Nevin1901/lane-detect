import numpy as np
import cv2 as cv
from mss import mss

if __name__ == "__main__":
    with mss() as sct:
        monitor = {
        "top": 0,
        "left": 0,
        "width": 1920,
        "height": 1080,
        }
        print(sct.grab(monitor))
