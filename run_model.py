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


def run(num_classes,learning_rate,width,depth,mini_batch_size):

	precision = accuracy = recall = f_score = np.array([])


	X_train,X_test,y_train,y_test,unknown_data = dp.load_data()
	X_train,X_test,y_train,y_test,unknown_data,dtype = dp.prepare_data(X_train,X_test,y_train,y_test,unknown_data)


	for _ in range(1):

		model = NN.Net1(num_classes,depth=depth,width=width).type(dtype)
		opt = optim.SGD(params=model.parameters(),lr=learning_rate,momentum=rp.m,nesterov=True)
		train_losses,test_losses = model.train_validate(X_train,y_train,X_test,y_test,opt,mini_batch_size,dtype)

		model = torch.load("Models/Best_Model.pkl")

		y_pred,_ = model.test(X_test)

		# Calculate metrics
		y_true = y_test.data.cpu().numpy()
		y_pred = y_pred.data.cpu().numpy()
		a,p,r,f = m.compute_metrics(y_true,y_pred)

		accuracy = np.append(accuracy,a)
		precision = np.append(precision,p)
		recall = np.append(recall,r)
		f_score = np.append(f_score,f)


	accuracy = np.mean(accuracy)
	precision = np.mean(precision)
	recall = np.mean(recall)
	f_score = np.mean(f_score)

	m.show_results(accuracy,precision,recall,f_score,num_classes,train_losses,test_losses)
	
	#g.generate_graph(model,X_train)
	
	fw.create_data_csv(learning_rate,depth,width,mini_batch_size,rp.m,len(test_losses)-10,accuracy)

	# Store unknown_data prediction 
	y_pred,_ = model.test(unknown_data)
	fw.store_prediction(y_pred.data.cpu().numpy())
