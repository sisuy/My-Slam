#! /usr/local/bin/python3
import cv2
import numpy as np
from frame import Frame
from skimage.measure import ransac 
from skimage.transform import EssentialMatrixTransform


W = 640
H = 400


class Slam:
    def __init__(self,W,H):
        self.W = W
        self.H = H
        self.points = []

        # Frames
        self.frames = []

    def add_points(points):
        points.append(points)

    def Transform(self,frame):
        frame = cv2.resize(frame,(W,H))
        return frame
        
    def process_Frame(self,frame):
        # Not allow none frame
        assert frame is not None    

        frame = self.Transform(frame)
        # frame = self.normalize(frame)

        frame =Frame(frame)
        
        # Add frames to Slam
        self.frames.append(frame)

        # Match two frames
        if len(self.frames) > 1:
            f1,f2 = self.frames[-2],self.frames[-1]
            match_pts1,match_pts2 = self.match_frames(f1,f2)

        return frame

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
                               random_state=None
                               )

        match_points1 = pts_ransac1[inliers]
        match_points2 = pts_ransac2[inliers]

        return [match_points1,match_points2]


        
    # Use to draw the match_points in a frame
    def plot_match(self,img1,img2, pts1, pts2,matches):
        ret = cv2.drawKeypoints(img2,pts2,None,(255,0,0),None)

        return ret


if __name__ == '__main__':
    # import video
    cap = cv2.VideoCapture('data/test_countryroad.mp4')
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total Frames: %d ' %(frames_count))

    slam = Slam(W,H)
    
    # Video Start
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is not False:
            print('*** Frame %d/%d ***' %(i, frames_count))
            frame = slam.process_Frame(frame)
            cv2.imshow('frame',frame.img)

            # Press q to exit the video 
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
        i += 1