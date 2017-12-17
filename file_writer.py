import pickle
import os
import csv

import runtime_parser as rp

def create_dir():

	if not rp.verbose:
		if not os.path.exists("Architecture"):
			os.makedirs("Architecture")

	
		if not os.path.exists("Results"):
			os.makedirs("Results")


	if not os.path.exists("Models"):
	    os.makedirs("Models")




def store_prediction(y_pred,ovr):

	if ovr:
		f = open('Data/TestData/predictions_class_ovr'+'.pkl', 'wb')
	else:
		f = open('Data/TestData/predictions_class'+'.pkl', 'wb')
	pickle.dump(y_pred + 1,f,protocol=2);
	f.close() 


def create_data_csv(learning_rate,learning_rate_decay,depth,width,mini_batch_size,accuracy):

	row = [learning_rate,learning_rate_decay,depth,width,mini_batch_size,accuracy]

	with open("Results/results.csv", 'a') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		if os.stat("Results/results_adam.csv").st_size == 0:
			row0 = ["Learning Rate","Rate Decay","Depth","Width","Batch_Size","Accuracy"]
			wr.writerow(row0)


		wr.writerow(row)


