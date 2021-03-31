import numpy as np
from scipy.ndimage import convolve
from PIL import Image


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    return np.load(path)


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """
    #Blue channel
    mask1 = np.zeros(bayerdata.shape)
    # print(bayerdata.shape)
    mask1[1::2, ::2] = 1

    b=bayerdata*mask1

    #Green channel
    mask2 = np.zeros(bayerdata.shape)
    mask2[::2, ::2] = 1
    mask2[1::2, 1::2] = 1

    g=bayerdata*mask2

    #Red channel
    mask3 = np.zeros(bayerdata.shape)
    mask3[::2, 1::2] = 1
    r=bayerdata*mask3

    return r,g,b
  


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """
    img=np.stack((r,g,b),axis=-1)
    return img





def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    weight_BR=1/4*np.array([[1,2,1],[2,4,2],[1,2,1]])     
    weight_G=1/4*np.array([[0,1,0],[1,4,1],[0,1,0]])

    r_output=convolve(r,weight_BR,mode='nearest')
    g_output=convolve(g,weight_G,mode='nearest')
    b_output=convolve(b,weight_BR,mode='nearest')

    output=np.stack((r_output,g_output,b_output),axis=-1)


    return output
    

    