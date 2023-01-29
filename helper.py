import numpy as np
import os


# Pose
def poseRt(R,t):
  Rt = np.eye(4)
  Rt[:3,:3] = R
  Rt[:3,3] = t
  return Rt

def toHomogeneous(points):
  ret = []
  for p in points:
    i = np.array([0,0,1])
    i[0] = p[0]
    i[1] = p[1]
    ret.append(i)
  return ret

def pixielToCamera(K,points):
  # points = toHomogeneous(points)
  ret = []
  for pt in points:
    p = np.array([0,0],dtype=float)
    p[0] = (pt[0] - K[0,2])/K[0,0]
    p[1] = (pt[1] - K[1,2])/K[1,1]
    ret.append(p)
  return np.array(ret)



