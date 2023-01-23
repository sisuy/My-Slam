import cv2

class Display2D():
    def __init__(self,path):
        self.cap = cv2.VideoCapture(path)
        print('--- Video information : Vido name: %s ,Total Frames: %d ---' %(path,self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        
    def annotate2D(self, frame1, frame2):
        assert frame1 is not None
        assert frame2 is not None
        assert len(frame1.match_points) == len(frame2.match_points)

        ret = frame2.img

        # Circle the keypoints of in two frames
        # TODO: use red circles to annotate the current frames'keypoints
        for i in frame1.match_points:
            ret = cv2.circle(ret,(int(i.pt[0]),int(i.pt[1])),2,(0,0,255))


        # TODO: use green circles to annotate the previous frame's keypoints
        for i in frame2.match_points:
            ret = cv2.circle(ret,(int(i.pt[0]),int(i.pt[1])),2,(0,255,0))

        # TODO: use blue line to annotate the track of the keypoints
        for i in range(len(frame1.match_points)):
            # distance1 = frame1.match_points[i].pt[0] - frame2.match_points[i].pt[0]
            # distance2 = frame1.match_points[i].pt[1] - frame2.match_points[i].pt[1]
            # print('distance x %d  --- distance y %d' %(distance1,distance2))
            ret = cv2.line(ret,
                          (int(round(frame1.match_points[i].pt[0])),int(round(frame1.match_points[i].pt[1]))),
                          (int(round(frame2.match_points[i].pt[0])),int(round(frame2.match_points[i].pt[1]))),
                          (255,0,0),
                          1)

        return ret
