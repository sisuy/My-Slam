import cv2
import pypangolin as pangolin
import OpenGL.GL as gl
from multiprocessing import Process,Queue
import numpy as np

class Display2D():
    def __init__(self,path):
        self.cap = cv2.VideoCapture(path)
        print('--- Video information : Vido name: %s ,Total Frames: %d ---' %(path,self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        
    def annotate2D(self, frame1, frame2):
        assert frame1 is not None
        assert frame2 is not None
        # assert len(frame1.match_points) == len(frame2.match_points)
        if frame1.match_points is None:
            return frame2.img

        ret = frame2.img

        # Use red circles to annotate the current frames'keypoints
        for i in frame1.match_points:
            ret = cv2.circle(ret,(int(i[0]),int(i[1])),2,(0,0,255))


        # Use green circles to annotate the previous frame's keypoints
        for i in frame2.match_points:
            ret = cv2.circle(ret,(int(i[0]),int(i[1])),2,(0,255,0))

        # Use blue line to annotate the track of the keypoints
        for i in range(len(frame1.match_points)):
            ret = cv2.line(ret,
                          (int(round(frame1.match_points[i][0])),int(round(frame1.match_points[i][1]))),
                          (int(round(frame2.match_points[i][0])),int(round(frame2.match_points[i][1]))),
                          (255,0,0), 1) 
        return ret 

class Display3D:
    def __init__(self,W,H):
        print("sadasdasdas init")
        self.state = None # Used to store camera state
        self.W = W
        self.H = H
        self.q = Queue() # q[0]: poses, q[1]: keypoints
        self.vp = Process(target=self.display_thread,args = (self.q,))
        self.vp.daemon = True
        self.vp.start()

    def display_thread(self,q):
        self.setup()
        while 1:
            self.run(q)

    def setup(self):
        # init pangolin GUI
        pangolin.CreateWindowAndBind('Viewer',self.W,self.H)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # gl.glEnable(gl.GL_BLEND)

        # Viewer, size = W*H
        self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(self.W, self.H, 420, 420, self.W//2, self.H//2, 0.2, 10000),
      pangolin.ModelViewLookAt(0, 0, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
      # pangolin.ModelViewLookAt(0, -10, -8, 
      #                          0,   0,  0, 
      #                          0,  -1,  0))

        # set bound and handler
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -self.W/self.H)
        self.handler = pangolin.Handler3D(self.scam)
        self.dcam.SetHandler(self.handler)
        self.dcam.Activate(self.scam)

    def run(self,q):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # load states from queue
        while not self.q.empty():
            self.state = q.get()
        pangolin.BindToContext('Viwer')


        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)
        
        # Draw previous camara with green color
        if len(self.state[0]) >= 1:
            gl.glColor3f(0,1,0)
            pangolin.DrawCameras(self.state[0][:-1])

            # Draw current camara with red color
            gl.glColor3f(1,0,0)
            pangolin.DrawCameras(self.state[0][-1:])

        # TODO: Draw keypoints
        if len(self.state[1]) >= 1:
            gl.glPointSize(4)
            gl.glColor3f(1,0,0)
            pangolin.DrawPoints(self.state[1],np.array(self.state[2])/256)

        pangolin.FinishFrame()
    
    def load_display(self,map):
        poses = []
        points = []
        colors = []
        # Add camara poses
        for f in map.frames:
            if f.pose is not None:
                poses.append(f.pose)
        
        # Add 3D points
        for p in map.points:
            points.append(p.location)
            colors.append(p.color)
            
        self.q.put([poses,points,colors])