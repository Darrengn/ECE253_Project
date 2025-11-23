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

def anhe(img, N_max = 100, K = 3, T = 5):
    """
    Runs adaptive neighborhood histogram equalization on the input image

    Args:
        img: 299x299x3 np array image in RGB format of image to run adjustment on
        N_max: max number of pixels in the adjustable neighborhood
        K: multiplicitive constant for standard deviation of neighborhood
        T: maximum pixel intensity difference from current pixel for neighborhood

    Returns:
        new_img: 299x299x3 np array image in RGB format of contrast adjusted image
    """
    # convert 3 channels into one intensity channel
    intensities = np.round(0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(int) 
    out_intens = np.zeros(intensities.shape)
    out_img = np.zeros_like(img)
    g_cdf = global_cdf(intensities)
    global_hist = 255 * g_cdf
    # loop through every pixel and get the new pixel intensity
    dirs = [(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)]
    for i in range(len(intensities)):
        for j in range(len(intensities[0])):
            cur_int = intensities[i,j]
            queue = [(i,j)]
            seen = {}
            count = 1
            neighborhood = [cur_int]
            hist = np.zeros(256)
            # find the adjustable neighborhood
            while queue and count < N_max:
                cur = queue.pop(0)
                if cur in seen:
                    continue
                for dir in dirs:
                    next = (cur[0] + dir[0], cur[1] + dir[1])
                    if next in seen or next[0] < 0 or next[0] >= len(intensities) or next[1] < 0 or next[1] >= len(intensities):
                        continue
                    next_int = intensities[next[0],next[1]]
                    if abs(next_int - cur_int) <= T:
                        queue.append(next)
                        neighborhood.append(next_int)
                        hist[next_int] += 1
                        count += 1
            # neighborhood is found
            hist /= count
            n_cdf = np.zeros(256)
            total = 0
            for k in range(len(hist)):
                total += hist[k]
                n_cdf[k] = total
            
            mean = round(np.mean(neighborhood))
            std = np.std(neighborhood)
            imin = max(round(global_hist[mean] - K * std), 0)
            imax = min(round(global_hist[mean] + K * std), 255)
            out_intens[i,j] = round(imin + (imax - imin) * n_cdf[cur_int])
            if cur_int == 0:
                ratio = 0
            else:
                ratio = out_intens[i,j] / cur_int
            out_img[i,j] = np.clip(ratio * img[i,j], 0, 255)
            
    return out_img

def contrast_adjust(img, delta = 50, lam = 1.4):
    intensities = np.round(0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2])/255
    mean = np.mean(intensities)
    gamma = np.sum(((intensities-mean))**2) / (299*299 - 1) * delta
    print(gamma)
    y = 1 + lam*gamma
    new_int = y * (intensities + gamma)
    ratio = np.where(intensities != 0.0, new_int / intensities, 0)
    print(np.max(ratio), np.min(ratio))
    
    new_img = np.clip(img * ratio[:, :, np.newaxis], 0, 255).astype(np.uint8)
    print(np.max(img * ratio[:, :, np.newaxis]))
    return new_img
    


# TODO: remove for testing
# anhe(np.zeros((299,299,3)))
# global_hist(np.ones((299,299), dtype=np.int8))