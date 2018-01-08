import numpy as np
from torch import optim
from torch.nn import functional as F
import torch
from torch.autograd import Variable


import NNet as NN
import data_preprocessing as dp
import metrics as m
import file_writer as fw
import graph_generator as g
import runtime_parser as rp


def run_full_net(num_classes,learning_rate,width,depth,mini_batch_size,ovr):

	precision = accuracy = recall = f_score = np.array([])

	best_test_losses = np.array([float('inf')])
	best_train_losses = np.array([float('inf')])

	X_train,X_test,y_train,y_test,unknown_data = dp.load_data()
	X_train,X_test,y_train,y_test,unknown_data,dtype = dp.prepare_data(X_train,X_test,y_train,y_test,unknown_data,ovr)


	for _ in range(10):

		model = NN.Net1(num_classes,depth=depth,width=width).type(dtype)
		opt = optim.SGD(params=model.parameters(),lr=learning_rate,momentum=rp.m,nesterov=True)
		train_losses,test_losses = model.train_validate(X_train,y_train,X_test,y_test,opt,mini_batch_size,"All",ovr,dtype)

		model = torch.load("Models/Best_Model_All.pkl")

		y_pred,_ = model.test(X_test)

		# Calculate metrics
		y_true = y_test.data.cpu().numpy()
		y_pred = y_pred.data.cpu().numpy()
		a,p,r,f = m.compute_metrics(y_true,y_pred)

		accuracy = np.append(accuracy,a)
		precision = np.append(precision,p)
		recall = np.append(recall,r)
		f_score = np.append(f_score,f)


		if np.mean(test_losses) < np.mean(best_test_losses):
			best_test_losses = test_losses
			best_train_losses = train_losses


	accuracy = np.mean(accuracy)
	precision = np.mean(precision)
	recall = np.mean(recall)
	f_score = np.mean(f_score)

	m.show_results(accuracy,precision,recall,f_score,num_classes,best_train_losses,best_test_losses,ovr)
	
	g.generate_graph(model,X_train)

	
	fw.create_data_csv(learning_rate,depth,width,mini_batch_size,rp.m,len(test_losses)-10,accuracy)

	# Store unknown_data prediction 
	y_pred,_ = model.test(unknown_data)
	fw.store_prediction(y_pred.data.cpu().numpy())



def run_ovr_nets(num_classes,learning_rate,width,depth,mini_batch_size,ovr):

	precision = accuracy = recall = f_score = np.array([])

	best_test_losses = np.array([float('inf')])
	best_train_losses = np.array([float('inf')])

	for _ in range(10):

		X_train,X_test,y_train,y_test,unknown_data = dp.load_data()
		X_train,X_test,training_sets,testing_sets,unknown_data,dtype = dp.prepare_data(X_train,X_test,y_train,y_test,
																					   unknown_data,ovr)	
		model0 = NN.Net1(num_classes,depth=depth,width=width).type(dtype)
		opt0 = optim.Adam(params=model0.parameters(),lr=learning_rate)

		model1 = NN.Net1(num_classes,depth=depth,width=width).type(dtype)
		opt1 = optim.Adam(params=model1.parameters(),lr=learning_rate)

		model2 = NN.Net1(num_classes,depth=depth,width=width).type(dtype)
		opt2 = optim.Adam(params=model2.parameters(),lr=learning_rate)

		model3 = NN.Net1(num_classes,depth=depth,width=width).type(dtype)
		opt3 = optim.Adam(params=model3.parameters(),lr=learning_rate)

		model4 = NN.Net1(num_classes,depth=depth,width=width).type(dtype)
		opt4 = optim.Adam(params=model4.parameters(),lr=learning_rate)


		train_losses0,test_losses0 = model0.train_validate(X_train,training_sets[0],X_test,testing_sets[0],opt0,mini_batch_size,"Car",ovr,dtype)
		train_losses1,test_losses1 = model1.train_validate(X_train,training_sets[1],X_test,testing_sets[1],opt1,mini_batch_size,"Dog",ovr,dtype)
		train_losses2,test_losses2 = model2.train_validate(X_train,training_sets[2],X_test,testing_sets[2],opt2,mini_batch_size,"Bicycle",ovr,dtype)
		train_losses3,test_losses3 = model3.train_validate(X_train,training_sets[3],X_test,testing_sets[3],opt3,mini_batch_size,"Motorbike",ovr,dtype)
		train_losses4,test_losses4 = model4.train_validate(X_train,training_sets[4],X_test,testing_sets[4],opt4,mini_batch_size,"Person",ovr,dtype)


		train_losses = [train_losses0,train_losses1,train_losses2,train_losses3,train_losses4]
		test_losses = [test_losses0,test_losses1,test_losses2,test_losses3,test_losses4]

		predictions = np.array([])
		y_pred = np.array([]).astype("int64")


		models = []

		models.append(torch.load("Models/Best_Model_Car.pkl"))
		models.append(torch.load("Models/Best_Model_Dog.pkl"))
		models.append(torch.load("Models/Best_Model_Bicycle.pkl"))
		models.append(torch.load("Models/Best_Model_Motorbike.pkl"))
		models.append(torch.load("Models/Best_Model_Person.pkl"))


		for model in models:
			_,out = model.test(X_test)			
			predictions = np.append(predictions,F.softmax(out,-1).data[:,1])
		
		predictions = np.reshape(predictions,(-1,5))
		
		y_pred = np.append(y_pred,np.argmax(predictions,axis=1))

		a,p,r,f = m.compute_metrics(y_test,y_pred)
			
		accuracy = np.append(accuracy,a)
		precision = np.append(precision,p)
		recall = np.append(recall,r)
		f_score = np.append(f_score,f)
	

	accuracy = np.mean(accuracy)
	precision = np.mean(precision)
	recall = np.mean(recall)
	f_score = np.mean(f_score)

	m.show_results(accuracy,precision,recall,f_score,5,train_losses,test_losses,ovr)
