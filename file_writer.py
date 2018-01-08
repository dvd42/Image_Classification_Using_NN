import pickle
import os
import csv

import runtime_parser as rp

def create_dir():

	if not os.path.exists("Architecture"):
		os.makedirs("Architecture")

	
	if not os.path.exists("Results"):
		os.makedirs("Results")


	if not os.path.exists("Models"):
	    os.makedirs("Models")




def store_prediction(y_pred):

	f = open('Data/TestData/predictions_class'+'.pkl', 'wb')
	pickle.dump(y_pred + 1,f,protocol=2);
	f.close() 


def create_data_csv(learning_rate,depth,width,mini_batch_size,momentum,epochs,accuracy):

	row = [learning_rate,momentum,"Y",depth,width,mini_batch_size,epochs,accuracy]

	with open("Results/results_adam.csv", 'a') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		if os.stat("Results/results_adam.csv").st_size == 0:
			row0 = ["Learning Rate","Momentum","Dropout","Depth","Width","Batch_Size","Epochs","Accuracy"]
			wr.writerow(row0)

		wr.writerow(row)


