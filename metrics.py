from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from matplotlib import pyplot as plt
import numpy as np

def compute_metrics(y_test,y_pred):

	y_true = y_test.data.cpu().numpy()
	y_pred = y_pred.data.cpu().numpy()

	#Store Metrics
	accuracy = accuracy_score(y_true,y_pred)
	precision = precision_score(y_true,y_pred,average="macro")
	recall = recall_score(y_true,y_pred,average="macro")
	f_score = f1_score(y_true,y_pred,average="macro")


	return accuracy,precision,recall,f_score



def show_results(accuracy,precision,recall,f_score,train_losses,test_losses,epochs,num_classes,y_pred,y_true,model):


	y_pred = y_pred.data.cpu().numpy()
	y_true = y_true.data.cpu().numpy()

	print(model)
	# Build Confusion Matrix
	confMat = np.zeros((num_classes,num_classes))    
	for pred in range(y_pred.size):
		confMat[y_true[pred],y_pred[pred]] +=1

	print(confMat)
	print("Car",y_true[y_true[y_pred==0] == 0].size)
	print("Dog",y_true[y_true[y_pred==1] == 1].size)
	print("Bicycle",y_true[y_true[y_pred==2] == 2].size)
	print("Motorbike",y_true[y_true[y_pred==3] == 3].size)
	print("Person",y_true[y_true[y_pred==4] == 4].size)

	print("Accuracy: %.2f\nPrecision: %.2f\nRecall: %.2f\nF1_score: %.2f\n" %(accuracy,precision,recall,f_score))

	plt.figure()
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.plot(np.arange(1,epochs+1),train_losses,'C1',label='Train error')
	plt.plot(np.arange(1,epochs+1),test_losses,'C2',label='Test error')
	plt.legend()
	plt.show()
