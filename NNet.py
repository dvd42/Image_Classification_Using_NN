import torch
from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import runtime_parser as rp


import data_preprocessing as dp
import metrics as m
import file_writer as fw


class Net1(nn.Module):
	def __init__(self,num_classes):
		super().__init__()

		#TODO play with network architecture implement pyramid shape perhaps

		self.h1 = nn.Linear(199,150)
		self.h2 = nn.Linear(150,100)
		self.h3 = nn.Linear(100,80)
		self.h4 = nn.Linear(80,50)
		self.h5 = nn.Linear(50,30)
		self.h6 = nn.Linear(30,10)
		self.h7 = nn.Linear(10,num_classes)


	def forward(self,x):


		x = F.leaky_relu(self.h1(x))
		x = F.leaky_relu(self.h2(x))
		x = F.dropout(x,p=0.2)
		x = F.leaky_relu(self.h3(x))
		x = F.leaky_relu(self.h4(x))
		x = F.leaky_relu(self.h5(x))
		x = F.dropout(x,p=0.5)
		x = F.leaky_relu(self.h6(x))
		x = F.leaky_relu(self.h7(x))
		
		return x

class Net2(nn.Module):
	def __init__(self,num_classes):
		super().__init__()

		self.h1 = nn.Linear(199,100)
		self.h2 = nn.Linear(100,50)
		self.h3 = nn.Linear(50,num_classes)
		
	
	def forward(self,x):

	
		x = F.leaky_relu(self.h1(x))
		x = F.leaky_relu(self.h2(x))
		x = F.leaky_relu(self.h3(x))
		
		return x

def train_validate(X_train,y_train,X_test,y_test,opt,model,n_class,ovr):


	train_losses = test_losses =  np.array([])

	stop = False
	best_loss = float('inf')
	no_improvement = 0
	epoch = 0
	mini_batch_size = rp.size
	criterion = nn.CrossEntropyLoss()


	while(not stop):

		batch_losses = np.array([])
		
		permutation = torch.randperm(X_train.size()[0])
		model.train()
		for i in range(0,X_train.size()[0],mini_batch_size):

			opt.zero_grad()

			index = permutation[i:i+mini_batch_size]
			mini_batch_x,mini_batch_y = X_train[index],y_train[index]


			#TODO balance weights
			#ones = torch.nonzero(mini_batch_y.data)

			# Train
			out = model(mini_batch_x)
			_,y_pred = out.max(1)

			loss = criterion(out,mini_batch_y) 
			loss =  criterion(out,mini_batch_y)


			batch_losses = np.append(batch_losses,loss.data[0])


			#Learn
			loss.backward()
			opt.step()

		train_losses = np.append(train_losses,np.mean(batch_losses))
		
		#Validate
		model.eval()
		out = model(X_test)
		_,y_pred = out.max(1)
		loss = criterion(out,y_test)
		test_losses = np.append(test_losses,loss.data[0])

		epoch += 1

		#adjust_learning_rate(opt,epoch)
				
		stop,no_improvement,best_loss = hara_kiri(best_loss,test_losses[-1],no_improvement)	

		if best_loss < test_losses[-1]:
			torch.save(model,"Models/Best_Model_" + n_class + ".pkl")

	

	return train_losses,test_losses


def test(new_data,model):

	model.eval()
	out = model(new_data)
	_,y_pred = out.max(1)
		

	return y_pred,out


def hara_kiri(best_loss,current_loss,no_improvement):

	if current_loss < best_loss:
		best_loss = current_loss
		no_improvement = 0

	else:
		no_improvement +=1

	return True if no_improvement == rp.tolerance else False,no_improvement,best_loss


def adjust_learning_rate(optimizer, epoch):
	
	"""
	Sets the learning rate to the initial lr decayed by 10 every 10 epochs
	"""

	lr = rp.lr * (0.1 ** (epoch // 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr





def run_full_net(num_classes,learning_rate,ovr):

	X_train,X_test,y_train,y_test,unknown_data = dp.load_data()
	X_train,X_test,y_train,y_test,unknown_data,dtype = dp.prepare_data(X_train,X_test,y_train,y_test,unknown_data,ovr)

	model = Net1(num_classes).type(dtype)
	print(model)

	opt = optim.Adamax(params=model.parameters(),lr=learning_rate)
	
	train_losses,test_losses = train_validate(X_train,y_train,X_test,y_test,opt,model,"All",ovr)

	model = torch.load("Models/Best_Model_All.pkl")

	y_pred,_ = test(X_test,model)


	# Calculate metrics
	y_true = y_test.data.cpu().numpy()
	y_pred = y_pred.data.cpu().numpy()
	accuracy,precision,recall,f_score = m.compute_metrics(y_true,y_pred)
	m.show_results(accuracy,precision,recall,f_score,num_classes,y_pred,y_true,train_losses,test_losses,ovr)

	
	# Store unknown_data prediction 
	y_pred,_ = test(unknown_data,model)
	fw.store_prediction(y_pred.data.cpu().numpy(),ovr)



def run_ovr_nets(num_classes,learning_rate,ovr):

 
	X_train,X_test,y_train,y_test,unknown_data = dp.load_data()
	X_train,X_test,training_sets,testing_sets,unknown_data,dtype = dp.prepare_data(X_train,X_test,y_train,y_test,unknown_data,ovr)	


	criterion = nn.CrossEntropyLoss()
	model0 = Net2(num_classes).type(dtype)
	opt0 = optim.Adamax(params=model0.parameters(),lr=learning_rate)

	model1 = Net2(num_classes).type(dtype)
	opt1 = optim.Adamax(params=model1.parameters(),lr=learning_rate)

	model2 = Net2(num_classes).type(dtype)
	opt2 = optim.Adamax(params=model2.parameters(),lr=learning_rate)

	model3 = Net2(num_classes).type(dtype)
	opt3 = optim.Adamax(params=model3.parameters(),lr=learning_rate)

	model4 = Net2(num_classes).type(dtype)
	opt4 = optim.Adamax(params=model4.parameters(),lr=learning_rate)

	print(model0)


	train_losses0,test_losses0 = train_validate(X_train,training_sets[0],X_test,testing_sets[0],opt0,model0,"Car",ovr)
	print("Car Model")
	train_losses1,test_losses1 = train_validate(X_train,training_sets[1],X_test,testing_sets[1],opt1,model1,"Dog",ovr)
	print("Dog Model")
	train_losses2,test_losses2 = train_validate(X_train,training_sets[2],X_test,testing_sets[2],opt2,model2,"Bicycle",ovr)
	print("Bicycle Model")
	train_losses3,test_losses3 = train_validate(X_train,training_sets[3],X_test,testing_sets[3],opt3,model3,"Motorbike",ovr)
	print("Motorbike Model")
	train_losses4,test_losses4 = train_validate(X_train,training_sets[4],X_test,testing_sets[4],opt4,model4,"Person",ovr)
	print("Person Model")


	train_losses = [train_losses0,train_losses1,train_losses2,train_losses3,train_losses4]
	test_losses = [test_losses0,test_losses1,test_losses2,test_losses3,test_losses4]

	predictions = np.array([])
	y_pred = np.array([]).astype("int64")

	model0 = torch.load("Models/Best_Model_Car.pkl")
	model1 = torch.load("Models/Best_Model_Dog.pkl")
	model2 = torch.load("Models/Best_Model_Bicycle.pkl")
	model3 = torch.load("Models/Best_Model_Motorbike.pkl")
	model4 = torch.load("Models/Best_Model_Person.pkl")

	_,out = test(X_test,model0)
	predictions = np.append(predictions,F.softmax(out,-1).data[:,1])
	_,out = test(X_test,model1)
	predictions = np.append(predictions,F.softmax(out,-1).data[:,1])
	_,out = test(X_test,model2)
	predictions = np.append(predictions,F.softmax(out,-1).data[:,1])
	_,out = test(X_test,model3)
	predictions = np.append(predictions,F.softmax(out,-1).data[:,1])
	_,out = test(X_test,model4)
	predictions = np.append(predictions,F.softmax(out,-1).data[:,1])

	predictions = np.reshape(predictions,(-1,5))


	y_pred = np.append(y_pred,np.argmax(predictions,axis=1))

	accuracy,precision,recall,f_score = m.compute_metrics(y_test,y_pred)
	m.show_results(accuracy,precision,recall,f_score,5,y_pred,y_test,train_losses,test_losses,ovr)
