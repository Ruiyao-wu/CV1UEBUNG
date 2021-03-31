from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def loadimg(path):
    """ Load image file

    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """

    img=plt.imread(path)
    #Normalization
    #PIL.Image.open will return unsigned integer values between 0 and 255.
    img *= 1.0/255
    
    return img




def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """

    xbeginn=(fsize[0]-1)/2
    xend=fsize[0]-2

    if xbeginn == 0:
        xend=xbeginn+1

    x=np.arange(-xbeginn,xend,1)         #递增matrix
    factor=1/(2*np.pi*(sigma**2))   #Gauss Smoothing Faktor
    #Gauss Kernel proportional to
    Gx=factor*np.exp(-np.power(x,2)/(sigma**2*2))   #separable Filter G=Gx*Gy

    #Normalization: Kern werte add to 1
    # G_norm= fx*fy
    sume=sum(Gx)
    fx=Gx/sume                      #fx--1x3 Filter
    fy=fx.reshape(fx.shape[0],1)   # fy--3x1 Filter
    #nxn Filter = fx*fy
    g=fx*fy

    return g



def binomial2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """

    W=fsize[0]
    H=fsize[1]
    x=H
    k=np.arange(0,x,1) 
    if  H==1:
        x=W
        k=np.arange(0,x,1) 
        k=k.reshape(k.shape[0],1)
       
    wk=binom(x-1,k)
    #Noramalization
    wl=sum(wk)
    wk=wk/wl

    if W!=1 and H!=1:
        wk=wk*wk.reshape(wk.shape[0],1)

    return wk



def downsample2(img1,f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """

    img=deepcopy(img1)

    #Filtering
    img=convolve(img,f,mode='mirror')

    #down Sample by a factor of 2
    img = img[:,range(0,img.shape[1],2)]
    img = img[range(0,img.shape[0],2),:]

    return img



def upsample2(img1, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """
    #Insert 0

    img = deepcopy(img1)

    shape=np.shape(img)  #row x column
    
    obj1=np.arange(1,shape[0],1)
    img=np.insert(img,obj1,0,axis=0) #insert 0 between row
    zero1=np.zeros(shape[1])
    img= np.row_stack((img,zero1)) #insert 0 in last row
    

    shape = np.shape(img) 
    obj2=np.arange(1,shape[1],1)
    img=np.insert(img,obj2,0,axis=1) #insert between column
    
    zero2=np.zeros(shape[0])
    img = np.column_stack((img,zero2)) #insert in last column

    #Filtering
    img=convolve(img,f,mode='mirror')*4

    return img


def gaussianpyramid(img1, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    img=deepcopy(img1)
    s=1
    # declare empty list 
    pyramid=[]
    pyramid.append(img)

    #nlevel = 6, 0=img, 1-5 pyramid

    while s < nlevel :
        img = downsample2(img, f)
        pyramid.append(img)
        s+=1
        
    return pyramid



    


def laplacianpyramid(gpyramid1, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    gpyramid=deepcopy(gpyramid1)

    pyramid=[]
    i=1
    level=len(gpyramid)   # level = 6
    while i < level:       # 1--6  l(0)--l(5)
        gexpand=upsample2(gpyramid[i],f)   # gexpand(i) 
        l=gpyramid[i-1]-gexpand          #l(0)=g(0)-gexpand(1) 
        pyramid.append(l)
        i+=1
    
    pyramid.append(gpyramid[level-1])  #l(6)=g(6)

    return lpyramid

    

def reconstructimage(lpyramid1, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """

    lpyramid=deepcopy(lpyramid1)
    gpramid=[]

    level=len(lpyramid)
    gpramid.append(lpyramid[level-1])
    i=1

    while i < level:
        gexpand=upsample2(gpramid[i-1],f)
        add=gexpand+lpyramid[level-1-i]
        gpramid.append(add)
        i+=1

    img_rescon = np.asarray(gpramid[level-1])
    np.clip(img_rescon,0,1 )



    return img_rescon 



def amplifyhighfreq(lpyramid, l0_factor=5, l1_factor=2):    
    #l0 2-3 Noise acceptable,sharpend full-resolution
    #wenn factor too big, rec_img will be edges Dectetion and noise too much und too dark
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """

    lpyramid[0]=lpyramid[0]*l0_factor
    lpyramid[1]=lpyramid[1]*l1_factor

    return lpyramid

    


def createcompositeimage(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
        normalized to [0,1]
    """
    i=0
    array=[]
    row=pyramid[0].shape[0]

    #read each level
    while i < len(pyramid):
        size=pyramid[i].shape
        Add_array=np.zeros((row,size[1]))
        Pyra=pyramid[i]
        for ii in range(0,size[0]): #row
            for jj in range(0,size[1]): #column
                if ii < Pyra.shape[0] :
                    Add_array[ii,jj]=Pyra[ii,jj]

        #Normalization each level
        Add_array *= 1.0/np.max(Add_array)
        if i==0:
            array=Add_array
        else:
            array=np.append(array,Add_array,axis=1)

        i+=1
    img_composite = np.asarray(array)

    return img_composite

