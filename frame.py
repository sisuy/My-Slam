import numpy as np
import cv2

class Frame():
    def __init__(self,img):
        # Initalize variable
        self.img = img 
        # compute keypoints and descriptors
        self.Add_keyPoints(img)
        # The point matches with last frame
        self.match_points = None
        # self.pose = np.eye(4)
        self.pose = np.eye(4)
        
   
    def Add_keyPoints(self,img): 
        orb = cv2.ORB_create(nfeatures=None,
                             scaleFactor =0.9,
                             nlevels=8,
                             edgeThreshold=40,
                             firstLevel=6,
                             patchSize=50
                             ) 

        # Find interesting key points
        img = np.mean(img,axis=2).astype(np.uint8)
        pts = cv2.goodFeaturesToTrack(img,3000,0.01,7)

        kps = [cv2.KeyPoint(x = i[0][0], y = i[0][1],size = 0) for i in pts]
        kps,des = orb.compute(img,kps)

        # add keypoints and descriptors
        self.keyPoints = kps
        self.descriptors = des
