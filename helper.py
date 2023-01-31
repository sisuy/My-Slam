import numpy as np

# Pose
def poseRt(R,t):
  Rt = np.eye(4)
  Rt[:3,:3] = R
  Rt[:3,3] = -t.T
  return Rt

def pixielToCamera(K,points):
  points = points.copy()
  points[:,0] = (points[:,0] - K[0,2])/K[0,0]
  points[:,1] = (points[:,1] - K[1,2])/K[1,1]
  return points

def extractColor(img,points):
  ret = []
  for i in points:
    ret.append(img[int(i[1]),int(i[0])])
  return np.array(ret,dtype=np.float64)