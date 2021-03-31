import matplotlib.pyplot as plt
import numpy as np



#
# Problem 1
#
from problem1 import *

def problem1():
    """Example code implementing the steps in problem 1"""
    # default valuesS
    fsize = (5, 5)
    sigma = 1.4
    nlevel = 6
    def showimage(img):
        plt.figure(dpi=150)
        plt.imshow(img, cmap="gray", interpolation="none")
        plt.axis("off")
        plt.show()
    #load image and build Gaussian pyramid
    img = loadimg("C:/Users/dreams/anaconda3/cv_uebung/assignment2/data/a2p1.png")
    fig = plt.figure(dpi=150)
    fig.add_subplot(131)
    showimage(img)

    img2=downsample2(img, gf)
    fig.add_subplot(132)
    showimage(img2)

    img3=upsample2(img, bf)
    fig.add_subplot(133)
    showimage(img3)
    gf = gauss2d(sigma, fsize)
    gpyramid = gaussianpyramid(img, nlevel, gf)
    showimage(createcompositeimage(gpyramid))

    # build Laplacian pyramid from Gaussian pyramid
    bf = binomial2d(fsize)
    lpyramid = laplacianpyramid(gpyramid, bf)

    # amplifiy high frequencies of Laplacian pyramid
    lpyramid_amp = amplifyhighfreq(lpyramid)
    showimage(createcompositeimage(lpyramid_amp))

    #reconstruct sharpened image from amplified Laplacian pyramid
    img_rec = reconstructimage(lpyramid_amp, bf)
    print(img_rec)
    print('-----------------')
    print(img)
    A=createcompositeimage((img, img_rec, img_rec - img))
    showimage(createcompositeimage((img, img_rec, img_rec - img)))





Problem 2

import problem2 as p2

def problem2():
    """Example code implementing the steps in Problem 2"""

    def show_images(ims, hw, title='', size=(8, 2)):
        assert ims.shape[0] < 10, "Too many images to display"
        n = ims.shape[0]
        
        # visualising the result
        fig = plt.figure(figsize=size)
        for i, im in enumerate(ims):
            fig.add_subplot(1, n, i + 1)
            plt.imshow(im.reshape(*hw), "gray")
            plt.axis("off")
        fig.suptitle(title)
        plt.show()


    # Load images
    imgs = p2.load_faces("C:/Users/dreams/anaconda3/cv_uebung/assignment2/data/yale_faces")
    y = p2.vectorize_images(imgs)
    #hw=imgs.shape[1:]
    hw = imgs.shape[0:2]
    print("Loaded array: ", y.shape)

    # Using 2 random images for testing
    test_face = y[0, :]
    test_face2 = y[-1, :]
    show_images(np.stack([test_face, test_face2], 0), hw,  title="Sample images")

    # Compute PCA
    mean_face, u, cumul_var = p2.compute_pca(y)
    # Compute PCA reconstruction
    # percentiles of total variance
    ps = [0.5, 0.75, 0.9, 0.95]
    ims = []
    for i, p in enumerate(ps):
        b = p2.basis(u, cumul_var, p)
        a = p2.compute_coefficients(test_face, mean_face, b)
        ims.append(p2.reconstruct_image(a, mean_face, b))

    show_images(np.stack(ims, 0), hw, title="PCA reconstruction")

    # fix some basis
    b = p2.basis(u, cumul_var, 0.95)

    # Image search
    top5 = p2.search(y, test_face, b, mean_face, 5)
    show_images(top5, hw, title="Image Search")

    # Interpolation
    ints = p2.interpolate(test_face2, test_face, b, mean_face, 5)
    show_images(ints, hw, title="Interpolation")


if __name__ == "__main__":
    problem1()
    problem2()

