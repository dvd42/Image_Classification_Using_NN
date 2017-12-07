import torch
from torch import autograd,nn,optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split




if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor
	print("Running Model using GPU")

else: 
 	dtype = torch.FloatTensor  
 	print("Running Model using CPU :(")


dataset = load_iris()

X = dataset.data.astype("float64")
y = dataset.target

#Add some noise
X = np.append(X,np.random.rand(10000,X.shape[1]),axis=0)
y = np.append(y,np.random.randint(3,size=10000))



X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,stratify=y)

num_classes = np.bincount(y).size
learning_rate = 0.001
epochs = 400
hidden_size = 2


train_losses = np.array([])
test_losses = np.array([])


X_train = autograd.Variable(torch.Tensor(X_train).type(dtype))
y_train = autograd.Variable(torch.IntTensor(y_train).type(dtype)).long()
X_test = autograd.Variable(torch.Tensor(X_test).type(dtype))
y_test = autograd.Variable(torch.IntTensor(y_test).type(dtype)).long()


class Net(nn.Module):
	def __init__(self,input_size,hidden_size,num_classes):
		super().__init__()
		self.h1 = nn.Linear(input_size,hidden_size)
		self.h2 = nn.Linear(hidden_size,hidden_size)
		self.h3 = nn.Linear(hidden_size,hidden_size)
		self.h4 = nn.Linear(hidden_size,hidden_size)
		self.h5 = nn.Linear(hidden_size,hidden_size)
		self.h6 = nn.Linear(hidden_size,hidden_size)
		self.h7 = nn.Linear(hidden_size,hidden_size)
		self.h8 = nn.Linear(hidden_size,num_classes)


	def forward(self,x):

		x = self.h1(x)
		x = F.relu(x)
		x = self.h2(x)
		x = F.dropout(x,p=0.2)
		x = F.relu(x)
		x = self.h3(x)
		x = F.relu(x)
		x = self.h4(x)
		x = F.dropout(x,p=0.3)
		x = F.relu(x)
		x = self.h5(x)
		x = F.relu(x)
		x = self.h6(x)
		x = F.dropout(x,p=0.5)
		x = F.relu(x)
		x = self.h7(x)
		x = F.relu(x)
		x = self.h8(x)
		x = F.softmax(x)	

		return x


model = Net(input_size=X_train.size()[1],hidden_size=hidden_size,num_classes=num_classes).type(dtype)
opt = optim.Adam(params=model.parameters(),lr=learning_rate)

#Speeds up the training when using GPUs
model.fastest = True

for epoch in range(epochs):

	# Train
	out = model(X_train)
	_, pred = out.max(1)
	loss = F.cross_entropy(out,y_train)
	train_losses = np.append(train_losses,loss.data[0])

	#Learn
	model.zero_grad()
	loss.backward()
	opt.step()

	# Test
	model.eval()
	out = model(X_test)
	_,pred = out.max(1)
	loss = F.cross_entropy(out,y_test)
	test_losses = np.append(test_losses,loss.data[0])


plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(np.arange(1,epochs+1),train_losses,'C1',label='Train error')
plt.plot(np.arange(1,epochs+1),test_losses,'C2',label='Test error')
plt.legend()
plt.show()
