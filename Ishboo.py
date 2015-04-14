#
# Imports
#

import numpy as np

class IshbooNeuralNetwork:

	# Class Members
	
	layerCount 	= 0	
	shape		= None
	weights		= []

	## Class Methods
	
	def __init__(self, layerSize):

		## Layer Info
		self.layerCount = len(layerSize) - 1
		self.shape		= layerSize	
		
		## Input/Output data from last run
		self._layerInput	= []
		self._layerOutput 	= []

		## Create the weight arrays	
		for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
			
			self.weights.append(np.random.normal(scale=0.1, size = (12, 11+1)))
			
	## 
	## Run method

	def Run(self, input):
		
		## run network based on the input data
		lnCases = input.shape[0]

		## clear out the previous intermediate value lists
		self._layerInput  = []
		self._layerOutput = []

		## run the net
		for index in range(self.layerCount):
			
			if index == 0:

				layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))

			else:
				
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))

		return self._layerOutput[-1].T

	## Activation functions
	def sgm(self, x, Derivative=False):

		if not Derivative:
			
			return 1 / (1+np.exp(-x))

		else:

			out = self.sgm(x)
			return out * (1-out)


##
## If run as a script, create a test object
##

if __name__ == "__main__":

	bpn = IshbooNeuralNetwork((2, 2, 1))
	print(bpn.shape)
	print(bpn.weights)
	
	lvInput = np.array([[0,0], [1,1], [-1, 0.5]])
	lvOutput = bpn.Run(lvInput)

	print("Input: {0}\nOutput: {1}".format(lvInput, lvOutput))
