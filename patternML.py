from numpy import exp, array, random, dot

class neural_net:
	def __init__(self):
		random.seed(10)
		# A single neuron will have 3 inputs and 1 output with a random weight
		self.weights = 2 * random.random((3,1)) - 1

	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	def train(self, inputs, outputs, num):
		for iteration in range(num):
			output = self.think(inputs)
			error = outputs - output
			adjustment = dot(inputs.T, error * output*(1-output))
			self.weights += adjustment

	def think(self, inputs):
		result = self.__sigmoid(dot(inputs, self.weights))
		return result



network = neural_net()

# Training data
inputs = array([[1,8,1], [1,0,1], [0,6,1]])
outputs = array([[1,3,0]]).T

# Training the neural network
network.train(inputs, outputs, 10000)

# Getting output from the neural network
print(network.think(array([1,0,0])))
