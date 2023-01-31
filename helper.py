import numpy as np

def add_ones(points):
  ret = []
  for p in points:
    a = np.array([0,0,1])
    a[0] = p[0]
    a[1] = p[1]
    ret.append(a)
  return np.array(ret)

def normalise(Kinv,points):
  ret = []
  for p in add_ones(points):
    p.shape = (3,1)
    a = np.dot(Kinv,p)
    ret.append(a.T[0])
  return np.array(ret)

def triangulate(Kinv,Rt,points):
  camera_pts = normalise(Kinv,points)
  print(camera_pts)
  RT_inv = np.linalg.inv(Rt)
  pts = []
  print('------------ test ---------------')
  for p in camera_pts:
    p.shape = (3,1)
    a = np.dot(p,RT_inv)
    print(a)
    pts.append(a)
  pts = np.array(pts).T

  pts = (pts/pts[3]).T
  return pts[:,:3]

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