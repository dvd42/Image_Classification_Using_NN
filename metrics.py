from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from matplotlib import pyplot as plt
import numpy as np

def compute_metrics(y_true,y_pred):

	#Store Metrics
	accuracy = accuracy_score(y_true,y_pred)
	precision = precision_score(y_true,y_pred,average="macro")
	recall = recall_score(y_true,y_pred,average="macro")
	f_score = f1_score(y_true,y_pred,average="macro")


	return accuracy,precision,recall,f_score



def show_results(accuracy,precision,recall,f_score,num_classes,y_pred,y_true,train_losses,test_losses,ovr):

	y_true = y_true.astype("int64")

	# Build Confusion Matrix
	confMat = np.zeros((num_classes,num_classes))    
	for pred in range(y_pred.size):
		confMat[y_true[pred],y_pred[pred]] +=1

	print(confMat)

	TP = np.diagonal(confMat)

	print("Car",TP[0])
	print("Dog",TP[1])
	print("Bicycle",TP[2])
	print("Motorbike",TP[3])
	print("Person",TP[4])

	print("Accuracy: %.2f\nPrecision: %.2f\nRecall: %.2f\nF1_score: %.2f\n" %(accuracy,precision,recall,f_score))

	if not ovr:
		plt.figure()
		plt.xlabel("Epoch")
		plt.ylabel("Losses")
		plt.plot(train_losses,'C1',label="Train Error")
		plt.plot(test_losses,'C2',label="Test Error")
		plt.legend()
		plt.show()

	else:
		plt.figure()
		plt.xlabel("Epoch")
		plt.ylabel("Losses")
		plt.plot(train_losses[0],'C0',label="Car Train Error")
		plt.plot(test_losses[0],'C1',label="Car Test Error")
		plt.plot(train_losses[1],'C2',label="Dog Train Error")
		plt.plot(test_losses[1],'C3',label="Dog Test Error")
		plt.plot(train_losses[2],'C4',label="Bicycle Train Error")
		plt.plot(test_losses[2],'C5',label="Bicycle Test Error")
		plt.plot(train_losses[3],'C6',label="Motorbike Train Error")
		plt.plot(test_losses[3],'C7',label="Motorbike Test Error")
		plt.plot(train_losses[4],'C8',label="Person Train Error")
		plt.plot(test_losses[4],'C9',label="Person Test Error")
		plt.legend()
		plt.show()