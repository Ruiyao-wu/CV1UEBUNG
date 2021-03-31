import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


# Plot 2D points
def displaypoints2d(points):
  plt.figure(0)
  plt.plot(points[0,:],points[1,:], '.b')
  plt.xlabel('Screen X')
  plt.ylabel('Screen Y')


# Plot 3D points
def displaypoints3d(points):
  fig = plt.figure(1)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(points[0,:], points[1,:], points[2,:], 'b')
  ax.set_xlabel("World X")
  ax.set_ylabel("World Y")
  ax.set_zlabel("World Z")


def cart2hom(points):
  """ Transforms from cartesian to homogeneous coordinates.

  Args:
    points: a np array of points in cartesian coordinates

  Returns:
    points_hom: a np array of points in homogeneous coordinates
  """

  shape=np.shape(points) #rowxcolumn
  column=shape[1]
  one=np.ones(column)
  points_hom= np.row_stack((points,one)) 
  return points_hom



def hom2cart(points):
  """ Transforms from homogeneous to cartesian coordinates.

  Args:
    points: a np array of points in homogenous coordinates 

  Returns:
    points_hom: a np array of points in cartesian coordinates
  """

  #homo Point 4x2904
  points_car=np.delete(points, 3, 0)
  return points_car



def gettranslation(v):
  """ Returns translation matrix T in homogeneous coordinates for translation by v.

  Args:
    v: 3d translation vector

  Returns:
    T: translation matrix in homogeneous coordinates
  """
  einheit=np.identity(3)
  plus_row=np.array([0,0,0,1])

  T_1=np.column_stack((einheit,v.T)) 
  T= np.row_stack((T_1,plus_row)) 

  return T



def getxrotation(d):
  """ Returns rotation matrix Rx in homogeneous coordinates for a rotation of d degrees around the x axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rx: rotation matrix
  """
  zeros=np.zeros(3)
  plus_row=np.array([0,0,0,1])
  Rx=np.array([[1,0,0],[0,np.cos(d*np.pi/180),-np.sin(d*np.pi/180)],[0,np.sin(d*np.pi/180),np.cos(d*np.pi/180)]])
  Rx_1=np.column_stack((Rx,zeros.T)) 
  Rx_homo= np.row_stack((Rx_1,plus_row)) 
  return Rx_homo



def getyrotation(d):
  """ Returns rotation matrix Ry in homogeneous coordinates for a rotation of d degrees around the y axis.

  Args:
    d: degrees of the rotation

  Returns:
    Ry: rotation matrix
  """
  zeros=np.zeros(3)
  plus_row=np.array([0,0,0,1])
  Ry=np.array([[np.cos(d*np.pi/180),0,np.sin(d*np.pi/180)],[0,1,0],[-np.sin(d*np.pi/180),0,np.cos(d*np.pi/180)]])
  Ry_1=np.column_stack((Ry,zeros.T)) 
  Ry_homo= np.row_stack((Ry_1,plus_row)) 
  return Ry_homo



def getzrotation(d):
  """ Returns rotation matrix Rz in homogeneous coordinates for a rotation of d degrees around the z axis.

  Args:
    d: degrees of the rotation

  Returns:
    Rz: rotation matrix
  """
  zeros=np.zeros(3)
  plus_row=np.array([0,0,0,1])

  Rz=np.array([[np.cos(d*np.pi/180),-np.sin(d*np.pi/180),0],[np.sin(d*np.pi/180),np.cos(d*np.pi/180),0],[0,0,1]])
  Rz_1=np.column_stack((Rz,zeros.T)) 
  Rz_homo= np.row_stack((Rz_1,plus_row)) 
  return Rz_homo


def getcentralprojection(principal, focal):
  """ Returns the (3 x 4) matrix L that projects homogeneous camera coordinates on homogeneous
  image coordinates depending on the principal point and focal length.
  
  Args:
    principal: the principal point, 2d vector
    focal: focal length

  Returns:
    L: central projection matrix
  """
  px=principal[0]
  py=principal[1]

  L=np.array([[focal,0,px,0],[0,focal,py,0],[0,0,1,0]])
  return L


def getfullprojection(T, Rx, Ry, Rz, L):
  """ Returns full projection matrix P and full extrinsic transformation matrix M.

  Args:
    T: translation matrix 4x4
    Rx: rotation matrix for rotation around the x-axis 4x4
    Ry: rotation matrix for rotation around the y-axis 4x4
    Rz: rotation matrix for rotation around the z-axis  4x4
    L: central projection matrix 3x4

  Returns:
    P: projection matrix
    M: matrix that summarizes extrinsic transformations
  """
  
  R1=Rz.dot(Rx)
  R=R1.dot(Ry)
  M=R.dot(T)

  P=L.dot(M)  #3x4 Matrix
  return P,M


def projectpoints(P, X):
  """ Apply full projection matrix P to 3D points X in cartesian coordinates.

  Args:
    P: projection matrix  3x4 Matrix
    X: 3d points in cartesian coordinates  3x2904 Marix in cart

  Returns:
    x: 2d points in cartesian coordinates
  """

  X_homo=cart2hom(X)
  x=P.dot(X_homo)
  return x


def loadpoints():
  """ Load 2D points from obj2d.npy.

  Returns:
    x: np array of points loaded from obj2d.npy
  """

  path='C:/Users/dreams/anaconda3/uebung/assignment1/data/obj2d.npy'

  return np.load(path)


def loadz():
  """ Load z-coordinates from zs.npy.

  Returns:
    z: np array containing the z-coordinates
  """

  path='C:/Users/dreams/anaconda3/uebung/assignment1/data/zs.npy'

  z=np.load(path)
  return z


def invertprojection(L, P2d, z):
  """
  Invert just the projection L of cartesian image coordinates P2d with z-coordinates z.

  Args:
    L: central projection 3x4 matrix
    P2d: 2d image coordinates of the projected points
    z: z-components of the homogeneous image coordinates

  Returns:
    P3d: 3d cartesian camera coordinates of the points
  """
  P3d_before=cart2hom(P2d)
  P3d_after=P3d_before*z
  L_cart=np.delete(L,3,1)
  L_inv=inv(L_cart)
  P3d=L_inv.dot(P3d_after)  #3x4 .* 4x2904 --- 3x2904
  return P3d


def inverttransformation(M, P3d):
  """ Invert just the model transformation in homogeneous coordinates
  for the 3D points P3d in cartesian coordinates.

  Args:
    M: matrix summarizing the extrinsic transformations 4x4 Matrix
    P3d: 3d points in cartesian coordinates 3x2904 Matix

  Returns:
    X: 3d points after the extrinsic transformations have been reverted 4x2904 in homo
  """

 
  M_inv=inv(M)  #cart to 4x2904 homo
  P3d_homo=cart2hom(P3d)
  X=M_inv.dot(P3d_homo)  #4x2904 in homo
  return X


def p3multiplecoice():
  '''
  Change the order of the transformations (translation and rotation).
  Check if they are commutative. Make a comment in your code.
  Return 0, 1 or 2:
  0: The transformations do not commute.
  1: Only rotations commute with each other.
  2: All transformations commute.
  '''
  #The transformations do not commute.
  print(":The transformations do not commute.")


  return 0