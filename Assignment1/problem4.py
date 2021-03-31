import math
from functools import partial
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve
conv2d = partial(convolve, mode="mirror")


def gauss2d(sigma, fsize):
  """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """
  # xbeginn=(fsize[0]-1)/2
  # ybeginn=(fsize[1]-1)/2
  # xend=fsize[0]-1
  # yend=fsize[1]-1

  # if xbeginn == 0:
  #    xend=xbeginn+1
  # elif ybeginn == 0:
  #    ybeginn=-0
  #    yend=ybeginn+1


  # x=np.arange(-xbeginn,xend,1)         #递增matrix
  # y=np.arange(-ybeginn,yend,1)
  # factor=1/(2*np.pi*(sigma**2))
  # G=factor*np.exp(-(np.power(x,2)+np.power(y,2))/(sigma**2*2))  #np.power  gegen Elmente .^
  # sume=sum(G)
  # g=G/sume
  # g=g.reshape(g.shape[0],1)   # g--3x1 Filter

  # return g
  m, n = fsize
  x = np.arange(-m / 2 + 0.5, m / 2)
  y = np.arange(-n / 2 + 0.5, n / 2)
  xx, yy = np.meshgrid(x, y, sparse=True)
  g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
  return g / np.sum(g)

   


def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """

  dx = np.expand_dims(np.array([0.5, 0, -0.5]),0)
  print('dx',dx)
  gy=gauss2d(0.9,[1,3]) 
  fx= gy * dx
  fy = fx.T
  return fx,fy



def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

  Ix=ndimage.convolve(I,fx)
  Iy=ndimage.convolve(I,fy)

  return Ix,Iy



def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

    # threshold bigger, then less Edges (more wert=0 set), best
    # 0.1: Shadow in the water not clear
    # 0.01: too much Rauchen
    # between 0.1 and 0.01 i chose 0.05 and have given up a little Edges in the water for smaller line 


  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """

  edges=np.sqrt(Ix**2 + Iy**2)

  for i in range(len(edges)):
    for j in range(len(edges[i])):
      if edges[i][j] < thr:
        edges[i][j] = 0



  return edges


def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """
  edges2 = np.ones_like(edges)
  #zero padding
  #padeedges = np.pad(edges,1)
  #edge orientation in [-90 90]
  angle=np.arctan(Iy/(Ix + 1e-24))
  #22.5  PI/8
  pi8 = math.pi / 8

  # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]
  for i in range(1,len(edges)-1):
    for j in range(1,len(edges[i])-1):
       if np.logical_or((angle[i,j] <= -3*pi8) ,(angle[i,j] > 3*pi8 )):
         bottom=edges[i+1][j]
         top=edges[i-1][j]
         if np.logical_or((edges[i][j] < bottom),(edges[i][j] < top)):
           edges2[i][j]=0
  

  

  # handle left-to-right edges: theta in (-22.5, 22.5]

  for i in range(1,len(edges)-1):
    for j in range(1,len(edges[i])-1):
      if np.logical_and((angle[i,j] > pi8) ,( angle[i,j] <= pi8 )):
        right=edges[i][j+1]
        left=edges[i][j-1]
        if (edges[i][j] < right) or (edges[i][j] < left):
          edges2[i][j]=0
        
        

  # handle bottomleft-to-topright edges: theta in (22.5, 67.5]

  for i in range(len(edges)-1):
    for j in range(len(edges[i])-1):
      if np.logical_and((angle[i,j] > pi8) ,( angle[i,j] <= 3*pi8 )):
        bottom=edges[i+1][j-1]
        top=edges[i-1][j+1]
        if (edges[i][j] < bottom) or (edges[i][j] < top):
          edges2[i][j]=0
         

  # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]

  for i in range(len(edges)-1):
    for j in range(len(edges[i])-1):
      if np.logical_and((angle[i,j] > -3*pi8) ,( angle[i,j] <= -pi8 )) :
        bottom=edges[i+1][j+1]
        top=edges[i-1][j-1]
        if (edges[i][j] < bottom) or (edges[i][j] < top): 
          edges2[i][j]=0
          
  return edges2*edges
    
