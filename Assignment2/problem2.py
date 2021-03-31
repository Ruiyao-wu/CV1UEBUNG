import numpy as np
import os.path
from PIL import Image
import matplotlib.pyplot as plt
from scipy import linalg


def load_faces(path, ext=".pgm"):
    """Load faces into an array (H, W,N),
    where N is the number of face images and
    H, W are height and width of the images.
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        imgs: (N,H, W) numpy array
    """
    
    #oswalk Tets
    #files ist Daten Name
    i=0
    for root, dirs, files in os.walk(path,topdown=False):
        for name in files:
            if ext in name:
                #print(os.path.join(root, name))
                img=plt.imread(os.path.join(root, name))
                if i == 0 :
                    img2=img
                    i+=1
                else:
                    img2=np.dstack((img2,img))

    return img2

      
            

def vectorize_images(imgs):
    """Turns an  array (H, W ,N),
    where N is the number of face images and
    H, W are height and width of the images into
    an (N, M) array where M=H*W is the image dimension.
    
    Args:
        imgs: (N, H, W) numpy array
    
    Returns:
        x: (N, M) numpy array
    """
    #imgs H*W  96*84=8064
    #n=760
    #Shape imgs H*W*N
    size=imgs.shape[2]
    i=0
    x1=[]
    while i < size:
        Array=imgs[:,:,i]   # 按维数取array
        Vector= Array.flatten()
        x1.append(Vector)

        i+=1

    x=np.row_stack(x1)

    return x


def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an numpy array with N M-dimensional features
    
    Returns:
        mean_face: (M,) numpy array representing the mean face
        u: (M, M) numpy array, bases with D principal components
        cumul_var: (N, ) numpy array, corresponding cumulative variance
    """

    N=X.shape[0]

    #Mean Face
    # x(N=720, M=H*W=8064)
    #Mean Face
    mean_face=np.sum(X, axis=0)/N

    #Covariance Matrix
    #C= (sum(x_n-x_mean)(x_n-x_mean).T/N)  N=760
    X_dach=X-mean_face  #X_dach  N*M 760*8064

    #Trans:X_dach  M*N 8064*760
    X_dach=X_dach.T
    u,sval,v=linalg.svd(X_dach)
    cumul_var=sval**2/N
    #### musterlosung
    for i in range(cumul_var.shape[0]-1):
        cumul_var[i+1] = cumul_var[i] + cumul_var[i+1]

    return mean_face,u,cumul_var




def basis(u, cumul_var, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) numpy array containing principal components.
        For example, i'th vector is u[:, i]
        cumul_var: (N, ) numpy array, variance along the principal components.
    
    Returns:
        v: (M, D) numpy array, contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """
    
    M = u.shape[0]
    i = 0
    sume = 0
    Grenze = p * cumul_var.sum()
    while i<M:
        sume = sume + cumul_var[i]
        i+=1
        if sume >= Grenze:
            break
        
    v=u[:, 0:i]

    return v



def compute_coefficients(face_image, mean_face, u):
    """Computes the coefficients of the face image with respect to
    the principal components u after projection.
    
    Args:
        face_image: (M, ) numpy array (M=h*w) of the face image a vector
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        a: (D, ) numpy array, containing the coefficients
    """
    
    diff = face_image - mean_face  #x_n-x_dach=sum(ai*ui),(1*M)

    a = np.matmul(diff,u)  #1*D

    return a 


def reconstruct_image(a, mean_face, u):
    """Reconstructs the face image with respect to
    the first D principal components u.
    
    Args:
        a: (D, ) numpy array containings the image coefficients w.r.t
        the principal components u
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (M, ) numpy array, projected vector of face_image on 
        principal components
    """
    image_out = mean_face +np.matmul(u,a)

    return image_out 
    


def compute_similarity(Y, x, u, mean_face):
    """Compute the similarity of an image x to the images in Y
    based on the cosine similarity.

    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) image we would like to retrieve
        u: (M, D) bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector

    Returns:
        sim: (N, ) numpy array containing the cosine similarity values
    """
    
    sim=[]
    a_x = compute_coefficients(x, mean_face, u)
    x_rescon = reconstruct_image(a_x, mean_face, u)
    x_norm=linalg.norm(x_rescon)  #x norm
    
    for i in range(0,len(Y)):
        face_image = Y[i,:]
        a_y = compute_coefficients(face_image, mean_face, u)
        y_rescon = reconstruct_image(a_y, mean_face, u)
        norm =  x_norm * linalg.norm(y_rescon)  # /||a||*||b||
        
        sim.append(np.dot(x_rescon,y_rescon)/norm)
    sim=np.asarray(sim)  #list to np array


    return sim  

    
def search(Y, x, u, mean_face, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) numpy array, image we would like to retrieve
        u: (M, D) numpy arrray, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M) numpy array containing the top_n most similar images
        sorted by similarity
    """
    sim=compute_similarity(Y, x, u, mean_face)
    index = np.argpartition(sim,-top_n)[-top_n:] #find top n max sim
    index = index[np.argsort(-sim[index])]       #sort the result
    y = []            #list 
    for i in range(0,len(index)):
        y.append(Y[index[i],:])
    Y=np.asarray(y)  #list 2 array
    print(Y)

    return Y



def interpolate(x1, x2, u, mean_face, n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (M, ) numpy array, the first image
        x2: (M, ) numpy array, the second image
        u: (M, D) numpy array, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        n: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate n equally-spaced points on a line
    
    Returns:
        Y: (n, M) numpy arrray, interpolated results.
        The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """
    y=[]
    #x1 Projection
    a1 = compute_coefficients(x1, mean_face, u)
    y1 = reconstruct_image(a1, mean_face, u)
    #x2 Projevtion
    a2 = compute_coefficients(x2, mean_face, u)
    y2 = reconstruct_image(a2, mean_face, u)

    #equally-spaced points in [0,1]
    interp = np.linspace(0, 1.0, num=n)

    #Poly. Interpolation

    for i in range(0,len(interp)):
        y.append(y1 + (y2-y1) * interp[i])

    Y=np.asarray(y)  #list 2 array

    return Y

    
