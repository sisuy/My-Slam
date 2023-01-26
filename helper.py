import numpy as np
import os


# Pose
def poseRt(R,t):
  Rt = np.eye(4)
  Rt[:3,:3] = R
  Rt[:3,3] = t
  return Rt