import numpy as np
import os


# Pose
def poseRt(R,t):
  if np.linalg.det(R) < 1:
    print("inverse")
    R = -R
    t = -t
  Rt = np.eye(4)
  Rt[:3,:3] = R
  Rt[:3,3] = t
  return Rt

def extractRt(E):
  # https://stackoverflow.com/questions/20614062/pose-from-fundamental-matrix-and-vice-versa
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
  U,d,Vt = np.linalg.svd(E)
  assert np.linalg.det(U) > 0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0: 
    R = np.dot(np.dot(U,W.T), Vt)
  t = U[:, 2]
  #Rt = np.concatenate((R, t.reshape(3,1)), axis=1)
  return poseRt(R, t)

# normalize coords
def normalize(Kinv, pts):
  ret = add_ones(pts).T
  ret = np.dot(Kinv, ret).T[:, 0:2]
  return ret

# going back to u,v coords
def denormalize(K, pt):
  ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
  #ret /= ret[2]
  return int(round(ret[0])), int(round(ret[1]))

def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)