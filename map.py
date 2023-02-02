# Map class is used to store the information of points and camara poses
class Map:
    def __init__(self):
        self.frames = []
        self.points = []
    
    # build Point object according to the information given by input, and add points into the map point list
    def add_points(self,location,colors):
        i = 0
        for p in location:
            self.points.append(Point(p,colors[i]))
            i += 1
        print('------ 3D: build %d points in this frame -------'%location.shape[0])


class Point:
    def __init__(self,location,color = None):
        self.location = location 
        self.color = color
    


