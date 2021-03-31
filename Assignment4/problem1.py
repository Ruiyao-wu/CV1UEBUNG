import numpy as np
from scipy.ndimage import convolve, maximum_filter
import scipy.signal


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """
    # 2nd derivatives
    #fx, fy = derivative_filters()
    img = convolve(img,gauss,mode='mirror')
    Ix = convolve(img,fx,mode='mirror')
    Iy = convolve(img,fy,mode='mirror')
    #Filter the image with the filters delta_x , delta_y
    I_xx = convolve(Ix,fx,mode='mirror')
    I_yy = convolve(Iy,fy,mode='mirror')
    I_xy = convolve(Ix,fy,mode='mirror')
    return I_xx, I_yy, I_xy



def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """

    det_H = (I_xx*I_yy - I_xy*I_xy)*(sigma**4)
    return det_H



def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """
    #find maximum in 5*5 Window
    data_max = maximum_filter(criterion, size=5,mode='mirror')
    #all the other equas 0
    criterion[criterion != data_max] = 0
    #find the interest points with thershold
    rows, cols = np.nonzero(criterion > threshold)
    print('rows',rows)
    criterion_thresh = np.logical_and(data_max > threshold,criterion >= data_max)
    mask = np.zeros_like(criterion_thresh)
    mask[5:-5,5:-5] = criterion_thresh[5:-5,5:-5] 
    rows2, cols2 = np.nonzero(mask)
    print('rows2',rows2)
    return rows2, cols2

