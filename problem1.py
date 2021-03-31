import numpy as np
import matplotlib.pyplot as plt

def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    plt.figure()
    plt.imshow(img)
    plt.title("Image")
    plt.show()



def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """

    np.save(path,img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    return np.load(path)


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """
    flt=np.fliplr(img)
    
    return flt

def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """
    fig=plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1)
    ax1.set_title("normal image")
    ax1.axis("off")
    ax2 = plt.subplot(122)
    ax2.imshow(img2)
    ax2.set_title("mirrored image")
    ax2.axis("off")
    plt.show()



    
