from torch import nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
	def __init__(self,input_size,num_classes):
		super().__init__()

		hidden_size = int(input_size * 0.75)

		self.h1 = nn.Linear(input_size,hidden_size)
		new_size = hidden_size
		self.h2 = nn.Linear(new_size,int(hidden_size * 0.75))
		new_size = int(hidden_size * 0.75)
		self.h3 = nn.Linear(new_size,int(new_size * 0.75))
		new_size = int(new_size * 0.75)
		self.h4 = nn.Linear(new_size,int(new_size * 0.75))
		new_size = int(new_size * 0.75)
		self.h5 = nn.Linear(new_size,int(new_size * 0.75))
		new_size = int(new_size * 0.75)
		self.h6 = nn.Linear(new_size,int(new_size * 0.75))
		new_size = int(new_size * 0.75)
		self.h7 = nn.Linear(new_size,num_classes)



	def forward(self,x):

		
	
		x = self.h1(x)
		x = F.leaky_relu(x)
		x = self.h2(x)
		x = F.leaky_relu(x)
		x = self.h3(x)
		x = F.leaky_relu(x)
		x = F.dropout(x,p=0.2)
		x = self.h4(x)
		x = F.leaky_relu(x)
		x = self.h5(x)
		x = F.leaky_relu(x)
		x = self.h6(x)
		x = F.leaky_relu(x)
		x = F.dropout(x,p=0.5)
		x = self.h7(x)
		x = F.softmax(x)
	
		return x



def train_test(X_train,y_train,X_test,y_test,mini_batch_size,opt,model):


	train_losses = test_losses =  np.array([])

	stop = False
	best_loss = float('inf')
	no_improvement = 0
	epochs = 0

	while(not stop):

		batch_losses = np.array([])
		start = 0
		end = mini_batch_size
	
		for batch in range(int(X_train.size()[0]/mini_batch_size)):

			# Train
			out = model(X_train[start:end])
			_,y_pred = out.max(1)
			loss = F.cross_entropy(out,y_train[start:end])
			batch_losses = np.append(batch_losses,loss.data[0])

			start = end
			end += mini_batch_size

			#Learn
			model.zero_grad()
			loss.backward()
			opt.step()

		train_losses = np.append(train_losses,np.mean(batch_losses))

		# Validate
		model.eval()
		out = model(X_test)
		_,y_pred = out.max(1)
		loss = F.cross_entropy(out,y_test)
		test_losses = np.append(test_losses,loss.data[0])

	
		stop,no_improvement,best_loss = hara_kiri(best_loss,train_losses[-1],no_improvement,10)

		epochs +=1
	
	model.eval()
	out = model(X_test)
	_,y_pred = out.max(1)
		
	return train_losses,test_losses,epochs,y_pred


	


def test(new_data,model):

	model.eval()
	out = model(new_data)
	_,y_pred = out.max(1)

	return y_pred + 1


def hara_kiri(best_loss,current_loss,no_improvement,tolerance):

	if current_loss < best_loss:
		best_loss = current_loss
		no_improvement = 0

	else:
		no_improvement +=1

	return True if no_improvement == tolerance else False,no_improvement,best_loss
