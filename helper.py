import numpy as np

# Pose
def poseRt(R,t):
  Rt = np.eye(4)
  Rt[:3,:3] = R
  Rt[:3,3] = t
  return Rt

def toHomogeneous(points):
  return np.concatenate((points,np.ones((points.shape[0],1))),axis=1)

def pixielToCamera(K,points):
  ret = []
  for pt in points:
    p = np.array([0,0],dtype=float)
    p[0] = (pt[0] - K[0,2])/K[0,0]
    p[1] = (pt[1] - K[1,2])/K[1,1]
    ret.append(p)
  return np.array(ret)

def extractColor(img,points):
  ret = []
  for i in points:
    ret.append(img[int(i[1]),int(i[0])])
  return np.array(ret,dtype=np.float64)

def triangulate(K,pose,points):
  # turn into camera coordinate
  pts_2d = []
  for pt in points:
    p = np.array([0,0,1],dtype=float)
    p[0] = (pt[0] - K[0,2])/K[0,0]
    p[1] = (pt[1] - K[1,2])/K[1,1]
    pts_2d.append(p)

  # TODO: (bugs)camera to world
  pts_4d = []
  for p in pts_2d:
    pts_4d.append(np.dot(p.T,pose))

  pts_3d = np.array(pts_4d).T
  pts_3d = pts_3d/pts_3d[3]
  print(pts_3d.T)

  return pts_3d.T[:,:3]


# def triangulate(K,pose,homos):
#   ret = []
# 
#   for p in homos.T:
#     p.shape = (1,3)
#     ret.append(np.dot(np.dot(p,K),pose))
#   ret = np.array(ret)
#   ret.shape = (ret.shape[2],ret.shape[0])
#   ret = (ret/ret[3]).T
#   print(ret)
#   return ret[:,:3]
    






