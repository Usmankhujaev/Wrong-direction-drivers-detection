from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared  
    def register(self, centroid):
        self.objects[self.nextObjectID]=centroid
        self.disappeared[self.nextObjectID]=0
        self.nextObjectID +=1
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
    def update(self, rects):
        if len(rects)==0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID]+=1
                if self.disappeared[objectID]>self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (xmin, ymin, xmax, ymax)) in enumerate(rects):
            mid_x = int((xmin+xmax) / 2)
            mid_y = int((ymin+ymax) / 2)
            inputCentroids[i] = (mid_x, mid_y)
        if len(self.objects)==0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])        
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)
                if D.shape[0] >= D.shape[1]:
                    for row in unusedRows:
                        objectID=objectIDs[row]
                        self.disappeared[objectID] += 1
                        if self.disappeared[objectID]> self.maxDisappeared:
                            self.deregister(objectID)
                        else:
                            for col in unusedCols:
                                self.register(inputCentroids[col])
        return self.objects