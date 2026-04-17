import cv2
import os

def load_image(path):
    return cv2.imread(path)

def save_image(path, img):
    cv2.imwrite(path, img)