from skimage import color, restoration
import numpy as np

def wiener_deconv(img):
    '''
    Performs unsupervised Wiener deconvolution on the input image. Splits color images into HSV channels and performs deconvolution on V.
    
    :param img: Image as np array RGB (H, W, 3)
    '''
    assert img.dtype == np.float32 or img.dtype == np.float64, "Input image must be float32 or float64"
    psf = np.ones((5,5), np.float32) / 25
    psf /= psf.sum()


    hsv = color.rgb2hsv(img)

    restored_v, _ = restoration.unsupervised_wiener(hsv[..., 2], psf)
    
    hsv[..., 2] = restored_v
    

    restored_rgb = color.hsv2rgb(hsv)

    return np.clip(restored_rgb, 0, 1)