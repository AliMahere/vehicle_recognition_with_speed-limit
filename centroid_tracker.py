from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker():

	"""
		CentroidTracker class is a calss for traking object 
	"""
	def __init__(self, max_disappeared=30):
		"""CentroidTracker class constructor 

		Args:
			maxDisappeared (int, optional): maxmum number of disapper in frames . Defaults to 25.
		"""
	
		self.next_object_ID = 0
		self.objects = OrderedDict()
		self.objectsArray = []
		self.disappeared = OrderedDict()

		self.max_disappeared = max_disappeared

	def register(self, centroid, rect):
		"""regester a new object to objects dict 

		Args:
			centroid (1D numpy array): array contain two values of type int for x and y of center of object
			rect ([type]): 
		"""


		self.objects[self.next_object_ID] = centroid,rect
		self.disappeared[self.next_object_ID] = 0
		self.next_object_ID += 1

	def deregister(self, object_ID):
		"""deregister object after it reatches the maxmuma number of disapper.
		It delests the opject id form objects dict and disappeared dict.

		Args:
			objectID (Integer): ID of object .
		"""

		del self.objects[object_ID]
		del self.disappeared[object_ID]
	
	def update(self, rects):
		"""this function check of if ther is a miisng objects to increment their disappeared
			count also check if we have reached the maximum number of consecutive frames a given object has been marked as missing. 
			If that is the case we need to remove it from the tracking systems 
			then regestert the new boxes the doesn't attached to and IDs 
		Args:
			rects ([numpy Arryay]): array contains an arryes that represinent object location 
			ex:  rects[
				[startX, startY, endX, endY, Cid]
				[startX, startY, endX, endY, Cid]
			]

		Returns:
			dict: contains everj object in the tracking system with it's new location 
			key : represnts ID of object 
			value : array of two numpy suparrays first array contains two values represents X, y of the centroid of object 
			second array constains 5 values represents startX, startY, endX, endY, Cid of object 
		"""

		if len(rects) == 0:

			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				if self.disappeared[objectID] > self.max_disappeared:
					self.deregister(objectID)

			return self.objects

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY, _ )) in enumerate(rects):

			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i],rects[i])
		else:

			objectIDs = list(self.objects.keys())

			
			objectCentroids = []
			for  value in self.objects.values():
				objectCentroids.append(value[0])


			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			rows = D.min(axis=1).argsort()

			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):

				if row in usedRows or col in usedCols:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col],rects[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			if D.shape[0] >= D.shape[1]:

				for row in unusedRows:

					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					if self.disappeared[objectID] > self.max_disappeared:
						self.deregister(objectID)

			else:
				for col in unusedCols:
					self.register(inputCentroids[col], rects[col])
		return self.objects

	def reset(self):
		"""calling this function will reset trcker to zero .
		"""
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()