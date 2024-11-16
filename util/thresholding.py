import cv2
import numpy as np
from scipy.ndimage import generic_filter


def apply_gaussian_filter(image, sigma=1.0):
    return cv2.GaussianBlur(image, (0, 0), sigma)
def calculate_local_statistics(image, window_size):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image
    """ Calculate local mean and standard deviation. """
    mean = cv2.boxFilter(image, cv2.CV_64F, (window_size, window_size))
    sqmean = cv2.sqrBoxFilter(image, cv2.CV_64F, (window_size, window_size))
    variance = sqmean - mean**2
    stddev = np.sqrt(variance)
    return mean, stddev

# def sauvola_threshold(image, window_size=15, P=0.5):
#     if len(image.shape) > 2 and image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         image = image
#     mean, stddev = calculate_local_statistics(image, window_size)
#     S_max = stddev.max()
#     threshold = mean * (1 + P * ((stddev / S_max) - 1))
#     return threshold

def niblack_threshold(image, window_size=15, P=-0.2):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image
    mean, stddev = calculate_local_statistics(image, window_size)
    threshold = mean + P * stddev
    return threshold

def nick_threshold(image, window_size=15, P=-0.2):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image
    mean, stddev = calculate_local_statistics(image, window_size)
    z = np.mean(image)
    MN = window_size**2
    sum_b_i_squared = generic_filter(image**2, np.sum, size=window_size)
    threshold = z + P * np.sqrt((sum_b_i_squared - z**2) / MN)
    return threshold

def apply_threshold(image, threshold):
    binary_image = image > threshold
    return binary_image.astype(np.uint8) 


def sauvola_threshold(image, window_size=15, k=0.5, R=128):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mean = cv2.blur(image, (window_size, window_size))
    mean_sq = cv2.blur(image ** 2, (window_size, window_size))
    stddev = np.sqrt(mean_sq - mean ** 2)

    # Sauvola formula
    threshold = mean * (1 + k * ((stddev / R) - 1))
    binary_image = (image > threshold).astype(np.uint8) * 255

    return binary_image