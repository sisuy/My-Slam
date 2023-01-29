#! /usr/local/bin/python3
import cv2
import numpy as np
from frame import Frame
from skimage.measure import ransac 
from skimage.transform import FundamentalMatrixTransform,EssentialMatrixTransform
from display import Display2D,Display3D
import sys
import pypangolin as pangolin
import OpenGL.GL as gl
from map import Map
import helper

W = 640
H = 400


class Slam:
    def __init__(self,W,H,F,intrinsic,extrinsic,path):
        self.F = F
        self.W = W
        self.H = H
        self.K = np.array([[F, 0, W//2],
               [0, F, H//2],
               [0, 0,    1]],dtype=np.float64)
        self.map = Map()
        self.display2D = Display2D(path)
        self.display3D = Display3D(self.W,self.H)

        # Camara Intrinsic and Camara extrinsic
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic


        # Frames
        # self.frames = []

    def add_points(points):
        points.append(points)

    def Transform(self,frame):
        frame = cv2.resize(frame,(W,H))
        return frame 
    
    def estimate_pose(self,frame,E): 
        U,D,VT = np.linalg.svd(E) 
        W = np.array([[0,-1,0], 
                      [1,0,0],
                      [0,0,1]])
        
        # u3 is U[:,2]
        T = U[:,2]
        # Rotation matrix
        R = np.dot(np.dot(U,W),VT)

        ret = np.eye(4)
        ret[:3,:3] = R
        ret[:3,3] = T.ravel() 
        return ret


    def process_frame(self,frame):
        # Not allow none frame
        assert frame is not None    
        f2 = Frame(self.Transform(frame),self.K)
        print('--- Found %d keypoints ---'% len(f2.keyPoints))

        # Match two frames
        if len(self.map.frames) > 1:
            f1 = self.map.frames[-1]
            f2,Rt = self.match_frames(f1,f2)

            f2.pose = np.dot(f1.pose,Rt)
            print('----- pose ---------')
            print(f2.pose)
            print('--------------------')

            # TODO: turn the pixel location of the keypoints into camera location(use the instrics of the cemera) r u sure?
            f2.homos = helper.pixielToCamera(self.K,f2.match_points)
            

            # TODO: Keypoints projection(point 4D)
            if f1.match_points is not None:
                points = cv2.triangulatePoints(f2.pose[:3,:],f1.pose[:3,:],f2.homos.T,f1.homos.T)
                points = (-points/points[3])[:3,:]
                colors = helper.extractColor(f2.img,f2.match_points)
                self.map.add_points(points.T,colors)

            f2.img = slam.display2D.annotate2D(f1,f2)
        # reset processed frame to slam frames list
        self.map.frames.append(f2)
        return f2

    def match_frames(self,f1,f2):
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck=True)
        matches = bf_matcher.match(f1.descriptors,f2.descriptors)

        # Find good features according to distance
        good = sorted(matches,key=lambda x:x.distance)[:70]

        # Use ransac to filt outliers
        pts_ransac1 = []
        pts_ransac2 = []

        pts1 = []
        pts2 = []

        for i in matches:
            pts1.append(f1.keyPoints[i.queryIdx])
            pts2.append(f2.keyPoints[i.trainIdx])

            pts_ransac1.append(f1.keyPoints[i.queryIdx].pt)
            pts_ransac2.append(f2.keyPoints[i.trainIdx].pt)

        # Convert into np.Array
        pts_ransac1 = np.array(pts_ransac1)
        pts_ransac2 = np.array(pts_ransac2)

        model, inliers =ransac(data=[pts_ransac1,pts_ransac2],
                               model_class=EssentialMatrixTransform,
                               min_samples=8,
                               residual_threshold=1,
                               max_trials=1000,
                               )


        match_points1 = pts_ransac1[inliers]
        match_points2 = pts_ransac2[inliers]

        # build cv2.KeyPoint list in each frame
        # f1.match_points = [cv2.KeyPoint(x = i[0], y = i[1], size = None) for i in match_points1]
        # f2.match_points = [cv2.KeyPoint(x = i[0], y = i[1], size = None) for i in match_points2]
        f1.match_points = match_points1
        f2.match_points = match_points2

        f1.homos = helper.pixielToCamera(self.K,match_points1)
        f2.homos = helper.pixielToCamera(self.K,match_points2)

        # extrac E from the matched points(get better performance than model.params)
        E = cv2.findEssentialMat(match_points1,match_points2,self.K,cv2.RANSAC)[0]

        # rebuild E, clean the noise 
        U,S,VT = np.linalg.svd(E)
        S = np.matrix([[S[0],0,0],
                    [0,S[1],0],
                    [0,0,0]])
        E = np.dot(np.dot(U,S),VT)

        # recover camera pose from two frames
        _,R,T,mask_match = cv2.recoverPose(E, match_points2, match_points1,self.K)

        # build camera pose from rotation matrix and translation matrix
        RT = helper.poseRt(R,T.T)

        return [f2,RT]

if __name__ == '__main__':
    F = sys.argv[1]
    path = sys.argv[2]
    slam = Slam(W,H,F,None,None,path)

    # Video Start
    i = 0
    frames_count = slam.display2D.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while(slam.display2D.cap.isOpened()):
        ret, frame = slam.display2D.cap.read()
        if ret is not False:
            print('*** Frame %d/%d ***' %(i, frames_count))
            frame = slam.process_frame(frame)
            slam.display3D.load_display(slam.map)
            cv2.imshow('frame',frame.img)


            # Press q to exit the video 
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
        i += 1
