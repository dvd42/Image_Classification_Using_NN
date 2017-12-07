import torch
from torch import autograd,nn,optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np



if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor
	print("Running Model using GPU")

else: 
 	dtype = torch.FloatTensor  
 	print("Running Model using CPU :(")


samples = 1000


train_batch_size = int(samples*0.75)
test_batch_size = int(samples*0.25)
input_size = 4

hidden_size = 4
num_classes = 4
learning_rate = 0.01
epochs = 300


train_losses = []
test_losses = []


in1 = autograd.Variable(torch.rand(train_batch_size,input_size).type(dtype))
in2 = autograd.Variable(torch.rand(test_batch_size,input_size).type(dtype))
train_target = autograd.Variable(torch.rand(train_batch_size).type(dtype)*num_classes).long()
test_target = autograd.Variable(torch.rand(test_batch_size).type(dtype)*num_classes).long()

class Net(nn.Module):
	def __init__(self,input_size,hidden_size,num_classes):
		super().__init__()
		self.h1 = nn.Linear(input_size,hidden_size)
		self.h2 = nn.Linear(hidden_size,num_classes)


	def forward(self,x):

		x = self.h1(x)
		x = F.dropout(x,p=0.2)
		x = F.tanh(x)
		x = self.h2(x)
		x = F.dropout(x,p=0.3)
		x = F.relu(x)
		x = F.relu(x)
		x = F.dropout(x,p=0.5)
		x = F.relu(x)
		x = F.softmax(x)
		
		return x


model = Net(input_size=input_size,hidden_size=hidden_size,num_classes=num_classes).type(dtype)
opt = optim.Adam(params=model.parameters(),lr=learning_rate)

#Speeds up the training when using GPUs
model.fastest = True

for epoch in range(epochs):

	# Train
	out = model(in1)
	_, pred1 = out.max(1)
	loss = F.nll_loss(out,train_target)
	train_losses.append(loss.data[0])

	#Learn
	model.zero_grad()
	loss.backward()
	opt.step()

	# Test
	out = model(in2)
	_,pred2 = out.max(1)
	loss = F.nll_loss(out,test_target)
	test_losses.append(loss.data[0])


plt.figure()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(np.arange(1,epochs+1),train_losses,'C1',label='Train error')
plt.plot(np.arange(1,epochs+1),test_losses,'C2',label='Test error')
plt.legend()
plt.show()










#model.zero_grad()
#model.parameters()

