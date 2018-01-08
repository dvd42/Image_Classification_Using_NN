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

	if not ovr:
		fig = plt.figure()
		fig.set_size_inches(10.5, 8.5)
		fig.set_facecolor("grey")
		plt.suptitle("Learning Rate = %s\nMomentum = %s\nDepth = %d\nWidth = %s \nBatch_size: %d"
		% (str(rp.lr),str(rp.m),rp.depth,str(rp.width),rp.size)
			,fontsize=11,style="oblique",fontweight="bold",bbox={"facecolor":"#cdc9c9","pad":5},ha="center")
		
		ax = fig.add_subplot(111)
		
		ax.text(1,0.1, "Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nF1_score: %.3f" %(accuracy,precision,recall,f_score), style='oblique',
        bbox={'facecolor':'#cdc9c9', 'pad':10},fontweight="bold",fontsize=11)
		
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Losses")
		ax.set_facecolor('black')

		ax.plot(train_losses,'C2',label="Train Error",marker=">")
		ax.plot(test_losses,'C2',label="Validation Error",marker="o")
		ax.legend().draggable()
		ax.axis([0,len(test_losses) + 3,0,max(max(test_losses),max(train_losses)) + 0.2])

		annotation = (np.argmin(test_losses) ,min(test_losses))

		ax.annotate('Best Model',color='white',fontweight='bold',fontsize=11,xy=annotation, xycoords='data',xytext=(annotation[0]-1,annotation[1] + 0.15),
			arrowprops=dict(facecolor='white', shrink=0.05),ha='center',va='bottom')
		

		if rp.verbose: 
			plt.show()
		
		else:
			plt.savefig('fig.png',facecolor=fig.get_facecolor(),dpu=100)

	else:
		fig = plt.figure()
		fig.set_size_inches(10.5, 8.5)
		fig.set_facecolor("grey")
		plt.suptitle("Learning Rate = %s\nBetas = %s\nDepth = %d\nWidth = %s \nBatch_size: %d"
		% (str(rp.lr),str(rp.b),rp.depth,str(rp.width),rp.size)
			,fontsize=11,style="oblique",fontweight="bold",bbox={"facecolor":"#cdc9c9","pad":5},ha="center")

		ax = fig.add_subplot(111)
		ax.text(1.5,0.1, "Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nF1_score: %.3f" %(accuracy,precision,recall,f_score), style='oblique',
        bbox={'facecolor':'#cdc9c9', 'pad':10},fontweight="bold",fontsize=11)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Losses")
		ax.set_facecolor('black')

		ax.plot(train_losses[0],'C0',label="Car Train Error",marker=">")
		ax.plot(test_losses[0],'C0',label="Car Validation Error",marker="o")
		ax.plot(train_losses[1],'C1',label="Dog Train Error",marker=">")
		ax.plot(test_losses[1],'C1',label="Dog Validation Error",marker="o")
		ax.plot(train_losses[2],'C2',label="Bicycle Train Error",marker=">")
		ax.plot(test_losses[2],'C2',label="Bicycle Validation Error",marker="o")
		ax.plot(train_losses[3],'C3',label="Motorbike Train Error",marker=">")
		ax.plot(test_losses[3],'C3',label="Motorbike Validation Error",marker="o")
		ax.plot(train_losses[4],'C4',label="Person Train Error",marker=">")
		ax.plot(test_losses[4],'C4',label="Person Validation Error",marker="o")
		ax.legend().draggable()

		ymax = 0
		for loss in test_losses:
			if max(loss) > ymax: ymax = max(loss)  


		ax.axis([0,len(max(test_losses,key=len)) + 3,0,ymax + 0.2])

		if rp.verbose:
			plt.show()
		else:
			plt.savefig('fig.png',facecolor=fig.get_facecolor(),dpi=100)