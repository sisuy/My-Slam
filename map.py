# Map class is used to store the information of points and camara poses
class Map:
    def __init__(self):
        self.frames = []
        self.points = []
    
    # build Point object according to the information given by input, and add points into the map point list
    def add_points(self,location):
        for p in location:
            self.points.append(Point(p))
        print('------ 3D: build %d points in this frame -------'%location.shape[0])


class Point:
    def __init__(self,location,color = None):
        self.location = location # location is a 3*1 array, which present a 3D location of a points
        # TODO: implement color of the points
    


