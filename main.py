import torch
from torch import optim
import numpy as np

import data_preprocessing as dp
import NNet as NN
import metrics as m
import file_writer as fw


num_classes = 5
learning_rate = 0.025
mini_batch_size = 256


X_train,X_test,y_train,y_test,unknown_data = dp.load_data()
X_train,X_test,y_train,y_test,unknown_data,dtype = dp.prepare_data(X_train,X_test,y_train,y_test,unknown_data)

		
model = NN.Net(input_size=X_train.size()[1],num_classes=num_classes).type(dtype)
opt = optim.Adamax(params=model.parameters(),lr=learning_rate)


train_losses,test_losses,epochs,y_pred = NN.train_test(X_train,y_train,X_test,y_test,mini_batch_size,opt,model)


# Calculate metrics
accuracy,precision,recall,f_score = m.compute_metrics(y_test,y_pred)
m.show_results(accuracy,precision,recall,f_score,train_losses,test_losses,epochs,num_classes,y_pred,y_test,model)


y_pred = NN.test(unknown_data,model)
fw.store_prediction(y_pred.data.cpu().numpy())
