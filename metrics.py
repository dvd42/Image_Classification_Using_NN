from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from matplotlib import pyplot as plt
import numpy as np

import runtime_parser as rp

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


	if not ovr:
		fig = plt.figure()
		fig.set_facecolor("grey")
		plt.suptitle("Learning Rate = %.2f\nMini_Batch_Size: %d\nTolerance: %d" % (rp.lr,rp.size,rp.tolerance)
			,fontsize=11,style="oblique",fontweight="bold",bbox={"facecolor":"white","alpha": 0.5,"pad":5},ha="center")

		ax = fig.add_subplot(111)
		ax.text(1,0.1, "Accuracy: %.2f\nPrecision: %.2f\nRecall: %.2f\nF1_score: %.2f" %(accuracy,precision,recall,f_score), style='oblique',
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},fontweight="bold",fontsize=11)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Losses")
		ax.set_facecolor('black')

		ax.plot(train_losses,'C2',label="Train Error",marker=">")
		ax.plot(test_losses,'C2',label="Test Error",marker="o")
		ax.legend()
		ax.axis([0,len(train_losses),0,max(test_losses) + 0.5])
		plt.show()

	else:
		fig = plt.figure()
		fig.set_facecolor("grey")
		plt.suptitle("Learning Rate = %.2f\nMini_Batch_Size: %d\nTolerance: %d" % (rp.lr,rp.size,rp.tolerance)
			,fontsize=11,style="oblique",fontweight="bold",bbox={"facecolor":"white","alpha": 0.5,"pad":5},ha="center")

		ax = fig.add_subplot(111)
		ax.text(1.5,0.1, "Accuracy: %.2f\nPrecision: %.2f\nRecall: %.2f\nF1_score: %.2f" %(accuracy,precision,recall,f_score), style='oblique',
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},fontweight="bold",fontsize=11)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Losses")
		ax.set_facecolor('black')

		ax.plot(train_losses[0],'C0',label="Car Train Error",marker=">")
		ax.plot(test_losses[0],'C0',label="Car Test Error",marker="o")
		ax.plot(train_losses[1],'C1',label="Dog Train Error",marker=">")
		ax.plot(test_losses[1],'C1',label="Dog Test Error",marker="o")
		ax.plot(train_losses[2],'C2',label="Bicycle Train Error",marker=">")
		ax.plot(test_losses[2],'C2',label="Bicycle Test Error",marker="o")
		ax.plot(train_losses[3],'C3',label="Motorbike Train Error",marker=">")
		ax.plot(test_losses[3],'C3',label="Motorbike Test Error",marker="o")
		ax.plot(train_losses[4],'C4',label="Person Train Error",marker=">")
		ax.plot(test_losses[4],'C4',label="Person Test Error",marker="o")
		ax.legend()
		# TODO adapt ymax
		ax.axis([0,len(max(train_losses,key=len)),0,1])
		plt.show()