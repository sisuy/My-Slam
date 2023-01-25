import cv2
import pypangolin as pangolin
import OpenGL.GL as gl
from multiprocessing import Process,Queue

class Display2D():
    def __init__(self,path):
        self.cap = cv2.VideoCapture(path)
        print('--- Video information : Vido name: %s ,Total Frames: %d ---' %(path,self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        
    def annotate2D(self, frame1, frame2):
        assert frame1 is not None
        assert frame2 is not None
        assert len(frame1.match_points) == len(frame2.match_points)

        ret = frame2.img

        # Use red circles to annotate the current frames'keypoints
        for i in frame1.match_points:
            ret = cv2.circle(ret,(int(i.pt[0]),int(i.pt[1])),2,(0,0,255))


        # Use green circles to annotate the previous frame's keypoints
        for i in frame2.match_points:
            ret = cv2.circle(ret,(int(i.pt[0]),int(i.pt[1])),2,(0,255,0))

        # Use blue line to annotate the track of the keypoints
        for i in range(len(frame1.match_points)):
            ret = cv2.line(ret,
                          (int(round(frame1.match_points[i].pt[0])),int(round(frame1.match_points[i].pt[1]))),
                          (int(round(frame2.match_points[i].pt[0])),int(round(frame2.match_points[i].pt[1]))),
                          (255,0,0), 1) 
        return ret 

class Display3D:
    def __init__(self,W,H):
        print("sadasdasdas init")
        self.W = W
        self.H = H
        self.q = Queue()
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
        gl.glEnable(gl.GL_BLEND)

        # Viewer, size = W*H
        self.scam = pangolin.OpenGlRenderState(pangolin.ProjectionMatrix(self.W,self.H,420,420,320,320,0.2,100),
                                          pangolin.ModelViewLookAt(2,0,2,0,0,0,pangolin.AxisY))

        # set bound and handler
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -self.W/self.H)
        self.handler = pangolin.Handler3D(self.scam)
        self.dcam.SetHandler(self.handler)
        self.dcam.Activate(self.scam)

    def run(self,q):
        pangolin.BindToContext('Viwer')


        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)
        
        pangolin.glDrawColouredCube()
        # draw something
        gl.glLineWidth(3)
        # TODO: paint camara pose 
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f ( 0.8,0,0 )
        gl.glVertex3f( -1,-1,-1 )
        gl.glVertex3f( 0,-1,-1 )
        gl.glColor3f( 0,0.8,0)
        gl.glVertex3f( -1,-1,-1 )
        gl.glVertex3f( -1,0,-1 )
        gl.glColor3f( 0.2,0.2,1)
        gl.glVertex3f( -1,-1,-1 )
        gl.glVertex3f( -1,-1,0 )
        gl.glEnd()
        pangolin.FinishFrame()

    def display(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)

        gl.glLineWidth(3)

        pangolin.FinishFrame()


