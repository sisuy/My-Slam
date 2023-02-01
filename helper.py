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

def triangulate(Kinv,pose1,pose2,pts1,pts2):
  # normalise the u,v(image) -> x,y(camera coordinate)
  pts1 = normalise(Kinv,pts1) 
  pts2 = normalise(Kinv,pts2) 

  # reference: ORB-SLAM2
  ret = np.zeros((pts1.shape[0], 4))
  for i, p in enumerate(zip(pts1, pts2)):
    A = np.zeros((4,4))
    A[0] = p[0][0] * pose1[2] - pose1[0]
    A[1] = p[0][1] * pose1[2] - pose1[1]
    A[2] = p[1][0] * pose2[2] - pose2[0]
    A[3] = p[1][1] * pose2[2] - pose2[1]
    _, _, vt = np.linalg.svd(A)
    ret[i] = vt[3]

  ret /= -ret[:,3:]
  return ret

# Pose
def poseRt(R,t):
  Rt = np.eye(4)
  Rt[:3,:3] = R
  Rt[:3,3] = -t.T
  return Rt

def extractColor(img,points):
  ret = []
  for i in points:
    ret.append(img[int(i[1]),int(i[0])])
  return np.array(ret,dtype=np.float64)