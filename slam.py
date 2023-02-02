#! /usr/local/bin/python3
import cv2
import numpy as np
from frame import Frame
from skimage.measure import ransac 
from skimage.transform import EssentialMatrixTransform
from display import Display2D,Display3D
import sys
from map import Map
import helper


class Slam:
    def __init__(self,W,H,F,path):
        # basic parameters
        self.F = F
        self.W = W
        self.H = H

        # camera matrix
        self.K = np.array([[F, 0, W/2],
                           [0, F, H/2],
                           [0, 0,    1]],dtype=np.float64)
        self.Kinv = np.linalg.inv(self.K)

        # visialization moudle
        self.map = Map()
        self.display2D = Display2D(path)
        self.display3D = Display3D(self.W,self.H)

    # transform the image according to what you need
    def Transform(self,img):
        frame = cv2.resize(img,(W,H))
        return frame 
    
    # process current frame
    def processFrame(self,frame):
        # Not allow none frame
        assert frame is not None    
        f2 = Frame(self.Transform(frame),self.K)

        # Match two frames
        if len(self.map.frames) > 1:
            f1 = self.map.frames[-1]
            
            # get rotation matrix and tranpose matrix 
            f2,Rt = self.match_frames(f1,f2)

            # calculate the essential matrix by previousPose * Rt
            f2.Rt = Rt
            f2.pose = np.dot(Rt,f1.pose)

            print('----- pose ---------')
            print(f2.pose)
            print('--------------------')

            pose1 = f1.pose.copy()
            pose2 = f2.pose.copy()

            # Triangulation
            m = helper.triangulate(self.Kinv,pose1,pose2,
                                   f1.matchPoints.copy(),f2.matchPoints.copy())
            helper.filter(self.K,f2.pose,m)
            m /= m[:,3:]
            
            # extrac the color of points
            colors = helper.extractColor(f2.img,f2.matchPoints)

            # add 3D points into map class
            self.map.add_points(m,colors)

            # annotate the 2D image, circle the feature points and show the track of the points
            f2.img = slam.display2D.annotate2D(f1,f2)

        # reset processed frame to slam frames list
        self.map.frames.append(f2)
        return f2

    def match_frames(self,f1,f2):
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck=True)
        matches = bf_matcher.match(f1.descriptors,f2.descriptors)

        # Use ransac to filt outliers
        ptsRansac1 = []
        ptsRansac2 = []

        pts1 = []
        pts2 = []

        for i in matches:
            pts1.append(f1.keyPoints[i.queryIdx])
            pts2.append(f2.keyPoints[i.trainIdx])

            ptsRansac1.append(f1.keyPoints[i.queryIdx].pt)
            ptsRansac2.append(f2.keyPoints[i.trainIdx].pt)

        # Convert into np.Array
        ptsRansac1 = np.array(ptsRansac1)
        ptsRansac2 = np.array(ptsRansac2)

        _, inliers =ransac(data=[ptsRansac1,ptsRansac2],
                               model_class=EssentialMatrixTransform,
                               min_samples=8,
                               residual_threshold=1,
                               max_trials=1000,
                               )


        matchPoints1 = ptsRansac1[inliers]
        matchPoints2 = ptsRansac2[inliers]

        # build cv2.KeyPoint list in each frame
        f1.matchPoints = matchPoints1
        f2.matchPoints = matchPoints2

        # extrac E from the matched points(get better performance than model.params)
        E = cv2.findEssentialMat(matchPoints2,matchPoints1,self.K,cv2.RANSAC)[0]
        E = helper.rebuildE(E)

        # recover camera pose from two frames
        _,R,T,mask = cv2.recoverPose(E, matchPoints1, matchPoints2,self.K)

        # build camera pose from rotation matrix and translation matrix
        RT = helper.poseRt(R,T)

        return [f2,RT]

if __name__ == '__main__':
    # set the width of the camera to 640 and the height of camera to 400
    W = 640
    H = 400
    F = sys.argv[1]
    videoPath = sys.argv[2]
    slam = Slam(W,H,F,videoPath)

    # Video Start
    i = 0
    framesCount = slam.display2D.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while(slam.display2D.cap.isOpened()):
        ret, frame = slam.display2D.cap.read()
        if ret is not False:
            print('*** Frame %d/%d ***' %(i, framesCount))
            frame = slam.processFrame(frame)

            # 3D display
            slam.display3D.loadDisplay(slam.map)

            # 2D display
            cv2.imshow('frame',frame.img)
            cv2.waitKey(1)
        i += 1
