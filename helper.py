import numpy as np

def rebuildE(E):
  # rebuild E, clean the noise 
  U,S,VT = np.linalg.svd(E)
  S = np.matrix([[S[0],0,0],
                 [0,S[1],0],
                 [0,0,0]])
  E = np.dot(np.dot(U,S),VT)
  return E

def addOnes(points):
  ret = np.concatenate((points,np.ones((points.shape[0],1))),axis = 1)
  return ret

def normalise(Kinv,points):
  ret = []
  for p in addOnes(points):
    p.shape = (3,1)
    a = np.dot(Kinv,p)
    ret.append(a.T[0])
  return np.array(ret)

def triangulate(Kinv,pose1,pose2,pts1,pts2):
  # reverte transpose matrix
  pose1[:,3] = -pose1[:,3]
  pose2[:,3] = -pose2[:,3]

  # normalise the u,v(image) -> x,y(camera coordinate)
  pts1 = normalise(Kinv,pts1) 
  pts2 = normalise(Kinv,pts2) 

  # reference: ORB-SLAM2 and github.com/geohotz/twitchslam
  ret = np.zeros((pts1.shape[0], 4))
  for i, p in enumerate(zip(pts1, pts2)):
    A = np.zeros((4,4))
    A[0] = p[0][0] * pose1[2] - pose1[0]
    A[1] = p[0][1] * pose1[2] - pose1[1]
    A[2] = p[1][0] * pose2[2] - pose2[0]
    A[3] = p[1][1] * pose2[2] - pose2[1]
    _, _, vt = np.linalg.svd(A)
    ret[i] = vt[3]

  return ret

def filter(K,pose,points):
  K = np.concatenate((K,np.zeros((3,1))),axis=1)
  Q = np.dot(K,pose)
  pts = []
  for p in points:
    p.shape = (4,1)
    p = np.dot(Q,p).T
    pts.append(p[0])

  pts = np.array(pts).T
  pts = (pts/pts[2]).T[:,:2]
  good = []

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