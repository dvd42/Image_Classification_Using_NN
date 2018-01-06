import torch
import numpy as np
import pickle
from torch.autograd import Variable

from sklearn.preprocessing import normalize
 
def load_data():
	f = open('Data/trainData/sift_sift_svm_traindata_K_199.pkl', 'rb')
	X_train = pickle.load(f,encoding='latin1')
	y_train = pickle.load(f,encoding='latin1')
	f.close()

	f = open('Data/valData/test_data_class_1.pkl', 'rb')
	X_test = pickle.load(f,encoding='latin1')
	y_test = pickle.load(f,encoding='latin1')
	f.close()

	f = open('Data/valData/test_data_class_2.pkl', 'rb')
	X_test = np.append(X_test,pickle.load(f,encoding='latin1'),axis=0)
	y_test = np.append(y_test,pickle.load(f,encoding='latin1'),axis=0)
	f.close()

	f = open('Data/valData/test_data_class_3.pkl', 'rb')
	X_test = np.append(X_test,pickle.load(f,encoding='latin1'),axis=0)
	y_test = np.append(y_test,pickle.load(f,encoding='latin1'),axis=0)
	f.close()

	f = open('Data/valData/test_data_class_4.pkl', 'rb')
	X_test = np.append(X_test,pickle.load(f,encoding='latin1'),axis=0)
	y_test = np.append(y_test,pickle.load(f,encoding='latin1'),axis=0)
	f.close()

	f = open('Data/valData/test_data_class_5.pkl', 'rb')
	X_test = np.append(X_test,pickle.load(f,encoding='latin1'),axis=0)
	y_test = np.append(y_test,pickle.load(f,encoding='latin1'),axis=0)
	f.close()

	y_train = y_train.reshape((y_train.shape[0],1))
	y_test = y_test.reshape((y_test.shape[0],1))


	f = open('Data/TestData/random_data_class.pkl', 'rb')
	unknown_data = pickle.load(f,encoding='latin1')
	f.close()



	return X_train,X_test,y_train-1,y_test-1,unknown_data



def prepare_data(X_train,X_test,y_train,y_test,unknown_data,ovr):


	if torch.cuda.is_available():
		dtype = torch.cuda.FloatTensor
	else: 
 		dtype = torch.FloatTensor

	training_set = np.append(X_train,y_train,axis=1)

	'''
	# Data augmentation 
	for i in range(1):
		augmented_data = np.array([])
		for i in range(5):
			index_train = y_train == i
			index_train = index_train.reshape((index_train.shape[0],))
			std = np.std(X_train[index_train],axis=0)
			new_data = X_train[index_train] + np.random.normal(loc=0.0,scale=std,size=X_train[index_train].shape)
			new_data = np.append(new_data,np.array([i]*X_train[index_train].shape[0]))
			augmented_data = np.append(augmented_data,new_data)

		training_set = np.append(training_set,augmented_data).reshape((-1,training_set.shape[1]))		
	'''
	
	

	test_set = np.append(X_test,y_test,axis=1)

	# Normalize and shuffle data
	training_set[:,:-1] = normalize(training_set[:,:-1])
	test_set[:,:-1] = normalize(test_set[:,:-1])
	unknown_data = normalize(unknown_data)

	np.random.shuffle(training_set)


	
 	# Prepare the data for ovr network architecture	
	if ovr:
		
		training_sets = []
		testing_set = []

		for i in range(5):
			y_train = np.copy(training_set[:,-1])
			index_train = y_train == i
			index_train = index_train.reshape((index_train.shape[0],))
			y_train[~index_train] = 0
			y_train[index_train] = 1

			y_test = np.copy(test_set[:,-1])
			index_test = y_test == i
			index_test = index_test.reshape((index_test.shape[0],))
			y_test[~index_test] = 0
			y_test[index_test] = 1


			y_train = Variable(torch.Tensor(y_train).type(dtype)).long()
			y_test = Variable(torch.Tensor(y_test).type(dtype)).long()
			training_sets.append(y_train)
			testing_set.append(y_test)


		X_train = Variable(torch.Tensor(training_set[:,:-1]).type(dtype))
		X_test = Variable(torch.Tensor(test_set[:,:-1]).type(dtype))
		unknown_data = Variable(torch.Tensor(unknown_data).type(dtype))

		return X_train,X_test,training_sets,testing_set,unknown_data,dtype


	X_train = Variable(torch.Tensor(training_set[:,:-1]).type(dtype))
	y_train = Variable(torch.Tensor(training_set[:,-1]).type(dtype)).long()
	X_test = Variable(torch.Tensor(test_set[:,:-1]).type(dtype))
	y_test = Variable(torch.Tensor(test_set[:,-1]).type(dtype)).long()
	unknown_data = Variable(torch.Tensor(unknown_data).type(dtype))


	return X_train,X_test,y_train,y_test,unknown_data,dtype