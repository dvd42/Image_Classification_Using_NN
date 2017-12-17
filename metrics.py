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





def show_results(accuracy,precision,recall,f_score,num_classes,train_losses,test_losses,ovr):

	#TODO: create annotation in plot at best error
	if not ovr:
		fig = plt.figure()
		fig.set_facecolor("grey")
		plt.suptitle("Learning Rate = %s\nRate Decay = %s\nDepth = %d\nWidth = %s \nBatch Size: %d\nTolerance: %d" 
			% (str(rp.lr),str(rp.lrd),rp.depth,str(rp.width),rp.size,rp.tolerance)
			,fontsize=11,style="oblique",fontweight="bold",bbox={"facecolor":"#cdc9c9","pad":5},ha="center")

		ax = fig.add_subplot(111)
		ax.text(1,0.1, "Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nF1_score: %.3f" %(accuracy,precision,recall,f_score), style='oblique',
        bbox={'facecolor':'#cdc9c9', 'pad':10},fontweight="bold",fontsize=11)

		ax.set_xlabel("Epoch")
		ax.set_ylabel("Losses")
		ax.set_facecolor('black')

		ax.plot(train_losses,'C2',label="Train Error",marker=">")
		ax.plot(test_losses,'C2',label="Test Error",marker="o")
		ax.legend().draggable()
		ax.axis([0,len(test_losses) + 3,0,max(max(test_losses),max(train_losses)) + 0.2])

		if rp.verbose:
			plt.show()

	else:
		fig = plt.figure()
		fig.set_facecolor("grey")
		plt.suptitle("Learning Rate = %s\nRate Decay = %s\nDepth = %d\nWidth = %s \nBatch Size: %d\nTolerance: %d" 
			% (str(rp.lr),str(rp.lrd),rp.depth,str(rp.width),rp.size,rp.tolerance)
			,fontsize=11,style="oblique",fontweight="bold",bbox={"facecolor":"#cdc9c9","pad":5},ha="center")

		ax = fig.add_subplot(111)
		ax.text(1.5,0.1, "Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nF1_score: %.3f" %(accuracy,precision,recall,f_score), style='oblique',
        bbox={'facecolor':'#cdc9c9', 'pad':10},fontweight="bold",fontsize=11)
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
		ax.legend().draggable()
		ymax = 0
		for loss in test_losses:
			if max(loss) > ymax: ymax = max(loss)  


		ax.axis([0,len(max(test_losses,key=len)) + 3,0,ymax + 0.2])

		if rp.verbose:
			plt.show()
