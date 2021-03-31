import numpy as np
import math
import random

class Problem2:
    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.

        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image

        Returns:
            distances: (n, m) numpy array, pairwise distances
        """
        m = features1.shape[1]
        n = features2.shape[1]
        #Distances initialize
        distances = np.zeros((n,m))
        for mm in range(0,m):
            for nn in range(0,n):
                diff = (features1[:,mm]-features2[:,nn])**2
                sumediff = np.sum(diff)
                distances[nn][mm] = sumediff

        ### loesung
        features1 = np.expand_dims(features1,axis=1)
        features2 = np.expand_dims(features2,axis=2)
        distances = np.sum((features1 - features2)**2,axis=0)
        return distances


    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.

        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix

        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """
        
        minindex = np.argmin(distances, axis=0)
        if p1.shape[0] <= p2.shape[0]:
            p2_small = p2[minindex,:]
            pairs = np.append(p1,p2_small,axis=-1)
        else: 
            pairs = np.append(p1[minindex,:],p2_small,axis=-1)
        return pairs


    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.

        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select

        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """
        #Randomly k pairs
        #mindimens = np.amin(p1.shape[0],p2.shape[0])
        #randindex1 = random.sample(range(0,mindimens-1),k)
        #loesung
        randindex = np.random.permutation(np.arange(p1.shape[0]))[:k]
        sample1 = p1[randindex,:]
        sample2 = p2[randindex,:]

        return sample1, sample2


        
    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """
        #ps1 = np.append(points,1,axis=-1)
        ####loesung
        ps = np.concatenate((points,np.ones((points.shape[0],1))),axis=1)
        #conditioning matrix
        ####loesung
        s = 0.5 * (np.max(np.absolute(points),axis=0))
        t = np.mean(points,axis=0)
        T = np.eye(3)
        T[0:2,2] = -t
        T[0:2,0:3] =  T[0:2,0:3] / np.expand_dims(s,axis=1)
        # u = T*x
        ps = ps.dot(T.T)
        return ps, T

    


    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices shoul be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.

        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2
        
        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """
        ####Loesung
        #A Matrix
        A= np.zeros((2*p1.shape[0],9))
        for i in range(0,2*p1.shape[0],2):
            z1 = p1[i//2]
            z2 = p2[i//2]
            
            # A[i][:3] = z1
            # A[i][6:] = -z1 * z2[0]
            # A[i+1][3:6] = -z1
            # A[i+1][6:] = z1 * z2[1]

            A[i][3:6] = z1
            A[i][6:] = -z1 * z2[1]
            A[i+1][:3] = -z1
            A[i+1][6:] = z1 * z2[0]

        _,_,VT = np.linalg.svd(A)
        V = VT.T

        HC = np.reshape(V[:,-1],(3,3))
        #nomalization, so that the roght bottom one is 1
        HC /= HC[2,2]
        H = np.linalg.inv(T2).dot(HC.dot(T1))
        H /= H[2,2]
        return H, HC


    


    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """
        points = np.concatenate((p,np.ones((p.shape[0],1))),axis=1).dot(H.T)
        #points_test = points[:,-1:] 
        points = points[:,:2] / (points[:,-1:] + 1e-24)
        return points
    

    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """
        p2_t =self.transform_pts(p1,H)
        p1_t =self.transform_pts(p2,np.linalg.inv(H))
        dist = np.linalg.norm(p2 - p2_t,axis=1)**2 + np.linalg.norm(p1 - p1_t,axis=1)**2
        return dist


    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance. 

        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold
        
        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
        inliers = pairs[(dist < threshold),:]
        return inliers.shape[0], inliers

    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """
        #n -- minimum number of required iterations
        #np.ceil 向上圆整
        n = int(np.ceil(math.log(1-z)/math.log(1-p**k)))
        return n

    


    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold
        
        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """
        pts1 = pairs[:,0:2]
        pts2 = pairs[:,2:4]
        best_H = None
        inlier = None
        max_inliers = -1

        for _ in range(n_iters-1):
            pts1_sub, pts2_sub =self.pick_samples(pts1,pts2,k)
            p1, T1 =self.condition_points(pts1_sub)
            p2, T2 =self.condition_points(pts2_sub)
            H, _ =self.compute_homography(p1,p2,T1,T2)
            dist =self.compute_homography_distance(H,pts1,pts2)

            num_inliers,inlier_temp =self.find_inliers(pairs,dist,threshold)
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H
                inlier = inlier_temp

        return best_H, max_inliers, inlier

    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
        pts1 = inliers[:,0:2]
        pts2 = inliers[:,2:4]
        p1, T1 =self.condition_points(pts1)
        p2, T2 =self.condition_points(pts2)
        H,_ =self.compute_homography(p1,p2,T1,T2)

        return H