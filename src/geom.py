import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg
# change to latex font


def log_map_gaussian(mean0, sigma0, mean1, sigma1):
    B = np.linalg.inv(linalg.sqrtm(sigma0)) @ linalg.sqrtm(linalg.sqrtm(sigma0) @ sigma1 @ linalg.sqrtm(sigma0)) @ np.linalg.inv(linalg.sqrtm(sigma0))
    A = B - np.eye(len(mean0))
    a = mean1 - mean0
    return A, a