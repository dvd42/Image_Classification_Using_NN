import pickle


def store_prediction(y_pred):

	f = open('Data/TestData/predictions_class'+'.pkl', 'wb')
	pickle.dump(y_pred,f,protocol=2);
	f.close()  