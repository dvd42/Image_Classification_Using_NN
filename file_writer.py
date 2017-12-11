import pickle
import os



def create_dir():

   
    if not os.path.exists("Models"):
        os.makedirs("Models")



def store_prediction(y_pred,ovr):

	if ovr:
		f = open('Data/TestData/predictions_class_ovr'+'.pkl', 'wb')
	else:
		f = open('Data/TestData/predictions_class'+'.pkl', 'wb')
	pickle.dump(y_pred + 1,f,protocol=2);
	f.close()  