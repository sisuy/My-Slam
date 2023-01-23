#! /usr/local/bin/python3
import cv2
import numpy as np
from frame import Frame
from skimage.measure import ransac 
from skimage.transform import EssentialMatrixTransform
from display import Display2D


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
        
    def process_frame(self,frame):
        # Not allow none frame
        assert frame is not None    

        frame = self.Transform(frame)
        # frame = self.normalize(frame)

        frame = Frame(frame)
        print('--- Found %d keypoints ---'% len(frame.keyPoints))
        
        # Add frames to Slam
        self.frames.append(frame)

        # Match two frames
        if len(self.frames) > 1:
            f1,f2 = self.frames[-2],self.frames[-1]
            match_pts1,match_pts2,f1,f2 = self.match_frames(f1,f2)
            f2.marked = display2D.annotate2D(f1,f2)

            # add processed frame to slam frames list
            self.frames[-1] = f2
            self.frames[-2] = f1
            frame = f2

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

        # build cv2.KeyPoint list in each frame
        f1.match_points = [cv2.KeyPoint(x = i[0], y = i[1], size = None) for i in match_points1]
        f2.match_points = [cv2.KeyPoint(x = i[0], y = i[1], size = None) for i in match_points2]
        return [match_points1,match_points2,f1,f2]


if __name__ == '__main__':
    # Initalize Display2D
    display2D = Display2D('data/test_countryroad.mp4')

    # import video
    slam = Slam(W,H)
    
    # Video Start
    i = 0
    frames_count = display2D.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while(display2D.cap.isOpened()):
        ret, frame = display2D.cap.read()
        if ret is not False:
            print('*** Frame %d/%d ***' %(i, frames_count))
            frame = slam.process_frame(frame)
            cv2.imshow('frame',frame.img)

            # Press q to exit the video 
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
        i += 1
