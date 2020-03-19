"""
Attempt to classify MNIST dataset using a simple feed-forward neural network with thre
"""

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

batch_size = 100 # batch_size hyper-parameter

# importing MNIST dataset
train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

# data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super().__init__()                             # Inherited from the parent class nn.Module
		self.fc1 = nn.Linear(784, hidden_size)  	   # 1st Fully-Connected Layer: 784 (input data) -> hidden_size (hidden node)
		self.relu = nn.ReLU()                          # Non-Linear ReLU Layer
		self.fc2 = nn.Linear(hidden_size, 10) 		   # 2nd Fully-Connected Layer: hidden_size (hidden node) -> 10 (output class)
    
	def forward(self, x):                              # Forward pass: stacking each layer together
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out
      
net = Net(input_size, hidden_size, num_classes) 
net.to(device) # load the net on GPU if present

net = Net(input_size, hidden_size, num_classes) 

use_cuda = True # set to False for loading net on cpu
if use_cuda and torch.cuda.is_available(): # load net on gpu (if present) and if flag is set to True 
	net.to(device)
	
# define function for the training process
def train(train_dataset, num_epochs, learning_rate):
	criterion = nn.CrossEntropyLoss() # criterion for evaluating network performance
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # using the Adam optimizer
	
	for epoch in range(num_epochs): # num_epochs full passes over the data
		for data in train_dataset:  # `data` is a batch of data
			X, y = data  # X is the batch of features, y is the batch of targets.
			X = X.view(-1,28*28) # change image from a matrix of 28 x 28 to a vector of size 784

			if use_cuda and torch.cuda.is_available(): # loading data on gpu (if present) and if flag is set to True
				X = X.cuda()
				y = y.cuda()        

			net.zero_grad()  # resets gradients to 0 before backprop begins
			outputs = net(X, input_size, hidden_size, num_classes)  # feed the data through the network
			loss = criterion(outputs, y)  # calc and grab the loss value
			loss.backward()  # apply this loss backwards thru the network's parameters
			optimizer.step()  # attempt to optimize weights to account for loss/gradients
			
			# observing loss at different instances
			if (i+1) % 100 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
			
# network hyper-parameters
hidden_size = 500
num_epochs = 5
learning_rate = 0.001

train(train_dataset, num_epochs, learning_rate) # training the network on train_dataset

# define function for testing accuracy of the network
def eval(test_dataset):
	correct = 0
	total = 0

	with torch.no_grad(): # no requirement of gradients for accuracy evaluation
		for data in test_dataset:
			X, y = data # get data for evaluation
			X = X.view(-1,28*28) # resizing data to appropriate size

		if use_cuda and torch.cuda.is_available(): # loading data on gpu (if present) and if flag is set to True
			X = X.cuda()
			y = y.cuda()

		outputs = net(X) # network outputs on test_dataset
		
		# accuracy calculation
		for idx, i in enumerate(outputs):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1
		print("Accuracy of the network on test_dataset: ", round(correct/total, 5)*100)

eval(test_dataset) # evaluate network performance
