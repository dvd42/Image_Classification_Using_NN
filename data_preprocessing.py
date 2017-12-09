import torch
import numpy as np
import pickle
from torch.autograd import Variable
from sklearn.decomposition import PCA


 
def load_data():
	f = open('Data/TrainData/sift_sift_svm_traindata_K_199.pkl', 'rb')
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



	return X_train,X_test,y_train,y_test,unknown_data


def prepare_data(X_train,X_test,y_train,y_test,unknown_data):

	# Shuffle Data
	training_set = np.append(X_train,y_train,axis=1)
	test_set = np.append(X_test,y_test,axis=1)

	# Data Augmentation
	flipped_images = np.fliplr(training_set[:,:-1])
	flipped_images = np.append(flipped_images,training_set[:,-1].reshape((-1,1)),axis=1)
	training_set = np.append(training_set,flipped_images,axis=0)
	
	np.random.shuffle(training_set)


	if torch.cuda.is_available():
		dtype = torch.cuda.FloatTensor
		print("Running Model using GPU")
	else: 
 		dtype = torch.FloatTensor
 		print("Running Model using CPU :(")


	X_train = training_set[:,:-1]
	y_train = training_set[:,-1]
	X_test = test_set[:,:-1]
	y_test = test_set[:,-1]

	"""
	# Get most important features
	pca = PCA(n_components=50)
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)
	unknown_data = pca.transform(unknown_data)
	"""

	
	print("Features",X_train.shape[1])
	
	X_train = Variable(torch.Tensor(X_train).type(dtype),requires_grad=True)
	y_train = Variable(torch.Tensor(y_train).type(dtype),requires_grad=True).long()
	X_test = Variable(torch.Tensor(X_test).type(dtype),requires_grad=True)
	y_test = Variable(torch.Tensor(y_test).type(dtype),requires_grad=True).long()
	unknown_data = Variable(torch.Tensor(unknown_data).type(dtype),requires_grad = True)


	return X_train,X_test,y_train-1,y_test-1,unknown_data,dtype