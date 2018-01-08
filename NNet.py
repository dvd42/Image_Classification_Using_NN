import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle

import runtime_parser as rp

class Net1(nn.Module):
	def __init__(self,num_classes,depth,width):
		super().__init__()

		self.depth = depth
		self.width = width

		fan_in = 199
		fan_out = int(199*self.width)
		for i in range(self.depth-1):
			self.__setattr__('h' + str(i), nn.Linear(fan_in,fan_out))
			init.orthogonal(self.__getattr__('h' + str(i)).weight,gain=init.calculate_gain('leaky_relu'))

			fan_in = fan_out
			fan_out = int(fan_in*self.width)

			
		self.__setattr__('dropout0.5',nn.Dropout(0.5,inplace=True))
		self.__setattr__('fc',nn.Linear(fan_in,num_classes))
		init.orthogonal(self.__getattr__('fc').weight,gain=init.calculate_gain('leaky_relu'))

	def forward(self,x):


		for i in range(self.depth - 1):
			x = F.leaky_relu(self.__getattr__('h' + str(i))(x))
			
					
		self.__getattr__('dropout0.5')(x)
		x = self.__getattr__('fc')(x)
		
		return x

	def train_validate(self,X_train,y_train,X_test,y_test,opt,mini_batch_size,n_class,ovr,dtype):


		train_losses = test_losses =  np.array([])

		stop = False
		best_loss = float('inf')
		no_improvement = 0
		epoch = 0

		if ovr:
			criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.2,1]).type(dtype))
		else:
			criterion = nn.CrossEntropyLoss()


		while(not stop):

			batch_losses = np.array([])
			
			permutation = torch.randperm(X_train.size()[0]).type(dtype).long()
			self.train()
			for i in range(0,X_train.size()[0],mini_batch_size):

				opt.zero_grad()

				index = permutation[i:i+mini_batch_size]
				mini_batch_x,mini_batch_y = X_train[index],y_train[index]


				# Train
				out = self(mini_batch_x)
				loss = criterion(out,mini_batch_y) 

				batch_losses = np.append(batch_losses,loss.data[0])

				#Learn
				loss.backward()
				opt.step()

			train_losses = np.append(train_losses,np.mean(batch_losses))

			adjust_learning_rate(opt,epoch)

			#Validate
			self.eval()
			out = self(X_test)
			loss = criterion(out,y_test)
			test_losses = np.append(test_losses,loss.data[0])

			epoch += 1

			stop,no_improvement,best_loss = self.hara_kiri(best_loss,test_losses[-1],no_improvement,10,n_class)	


		return train_losses,test_losses


	def test(self,new_data):

		self.eval()
		out = self(new_data)
		_,y_pred = out.max(1)

		return y_pred,out


	def hara_kiri(self,best_loss,current_loss,no_improvement,tolerance,n_class):

		if current_loss < best_loss:
			torch.save(self,"Models/Best_Model_" + n_class + ".pkl")
			best_loss = current_loss
			no_improvement = 0

		else:
			no_improvement +=1

		return True if no_improvement == tolerance else False,no_improvement,best_loss


def adjust_learning_rate(optimizer,epoch):

	"""
	Sets the learning rate to the initial lr decayed by 0.8 every 10 epochs
	"""

	lr = rp.lr * (0.8 ** (epoch // 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
