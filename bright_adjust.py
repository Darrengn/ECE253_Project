import numpy as np
import matplotlib.pyplot as plt
def global_cdf(intensities):
    """
    creates a global cdf from a one channel intensity image

    Args:
        intensities: 299x299 np array image of intensities get the global histogram of

    Returns:
        new_img: 256 length array where each entry at index i represents the global CDF(i)
    """
    pdf = np.zeros(256, dtype=float)
    cdf = np.zeros(256, dtype=float)
    pixel_count = len(intensities) * len(intensities[0])
    for i in range(len(intensities)):
        for j in range(len(intensities[0])):
            pdf[intensities[i,j]] += 1
    pdf /= pixel_count
    total = 0
    for i in range(len(pdf)):
        total += pdf[i]
        cdf[i] = total
    return cdf

def anhe(img, N_max = 100):
    """
    Runs adaptive neighborhood histogram equalization on the input image

    Args:
        img: 299x299x3 np array image in RGB format of image to run adjustment on
        N_max: max number of pixels in the adjustable neighborhood

    Returns:
        new_img: 299x299x3 np array image in RGB format of contrast adjusted image
    """
    # convert 3 channels into one intensity channel
    intensities = np.round(0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(int) 
    out_intens = np.zeros(intensities.shape)
    g_cdf = global_cdf(intensities)
    global_hist = 255 * g_cdf
    # loop through every pixel and get the new pixel intensity
    for i in range(len(intensities)):
        for j in range(len(intensities[0])):
            cur_int = intensities[i,j]
            queue = [(i,j)]
            seen = {}
            count = 0
            # find the adjustable neighborhood

# TODO: remove for testing
# anhe(np.zeros((299,299,3)))
# global_hist(np.ones((299,299), dtype=np.int8))