#! /usr/local/bin/python3
import cv2
import numpy as np
    
if __name__ == '__main__':
    # import video
    cap = cv2.VideoCapture('data/test_countryroad.mp4')
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total Frames: %d ' %(frames_count))
    
    # Video Start
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is not False:
            print('*** Frame %d/%d ***' %(i, frames_count))
            cv2.imshow('frame',frame)

            # Press q to exit the video 
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
        i += 1
