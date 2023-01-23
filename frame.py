import numpy as np
import cv2

class Frame():
    def __init__(self,img):
        # Initalize variable
        self.img = img 
        # marked img
        self.marked = None 
        # compute keypoints and descriptors
        self.Add_keyPoints(img)
        # The point matches with last frame
        self.match_points = None

        
   
    def Add_keyPoints(self,img): 
        orb = cv2.ORB_create(nfeatures=None,
                             scaleFactor =1.2,
                             nlevels=6,
                             edgeThreshold=40,
                             firstLevel=5,
                             patchSize=50
                             ) 

        # Find interesting key points
        img = np.mean(img,axis=2).astype(np.uint8)
        pts = cv2.goodFeaturesToTrack(img,1000,0.1,10)

        kps = [cv2.KeyPoint(x = i[0][0], y = i[0][1],size = 500) for i in pts]
        kps,des = orb.compute(img,kps)

        # add keypoints and descriptors
        self.keyPoints = kps
        self.descriptors = des
