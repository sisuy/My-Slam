import numpy as np
import cv2
import helper

class Frame():
    def __init__(self,img,K):
        # Initalize variable
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.Rt = np.zeros((3,4))

        self.img = img 
        # compute keypoints and descriptors
        self.Add_keyPoints(img)
        # The point matches with last frame
        self.matchPoints = None
        # init pose
        self.pose = np.eye(4)
        self.homos = None

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
        # kps = np.array([(int(kp.pt[0]), int(kp.pt[1])) for kp in kps])

        self.keyPoints = kps
        self.descriptors = des
