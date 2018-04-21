import torch
import json
import sys
import numpy as np
import pickle
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random
import os

train_dir = "/home/cse/btech/cs1150254/Desktop/ml/data/train/"
test_dir = "/home/cse/btech/cs1150254/Desktop/ml/data/test/"
out_dir = "/home/cse/btech/cs1150254/Desktop/ml/output/"
num_clus = 20
num_exam = 5000
num_feat = 784

def read_npy(directory):
    iterr=0
    label = []
    files = os.listdir(directory)
    for file_name in files:
        numpy_file = np.load(directory + file_name)
        if iterr==0:
            data = numpy_file
        else:
            data = np.concatenate((data,numpy_file))
        label.append(file_name[:-4])
        iterr+=1
    return data,label

train_data,file_labels = read_npy(train_dir)
test_data,_ = read_npy(test_dir)
actual_label = []
for i in range(len(train_data)):
    actual_label.append(i // num_exam)

class NEURAL_MODEL(torch.nn.Module) :
    def __init__(self,features,hidden_units,output_layer):
        super(NEURAL_MODEL,self).__init__()
        self.hidden = nn.Linear(features,hidden_units)
        self.predict = nn.Linear(hidden_units,output_layer)
        # self.hidden_dim = hidden_units
        # self. = num_layers
        # self.lstm = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        # self.linearOut = nn.Linear(2*hidden_dim,num_classes)
    def forward(self,inputs):
        lin_hidden_out = self.hidden(inputs)
        nl_hidden_out = F.sigmoid(lin_hidden_out)
        lin_out = self.predict(nl_hidden_out)
        x = F.log_softmax(lin_out)
        return x


def vectorize(document):
    global word_to_idx
    input_data = [word_to_idx[word] for word in document.split()]
    return input_data

def batchify(label_data,start_index):
    label_batch = []
    data_batch = []
    for i in range(start_index,start_index+batch_size):
        label_batch.append(label_data[i][0])
        data_batch.append(label_data[i][1])
    data_batch = np.asarray(data_batch,dtype="int")
    return (label_batch,data_batch)


hidden_dim = 2000
model = NEURAL_MODEL(num_feat,hidden_dim,num_clus)
model = model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 150
batch_size = 2000
num_batch = len(train_data) // batch_size

data_label_pair = []
for i in range(len(train_data)):
    data_label_pair.append((actual_label[i],train_data[i]))


for i in range(epochs):
    loss_sum = 0
    total_acc = 0.0
    random.shuffle(data_label_pair)
    for iterr in range(num_batch):
        label_batch,batch_data = batchify(data_label_pair,iterr*batch_size)
        batch_data = Variable(torch.FloatTensor(batch_data).cuda())
        target_data = Variable(torch.LongTensor(label_batch).cuda())
        class_pred = model(batch_data)
        model.zero_grad()
        loss = loss_function(class_pred,target_data)
        loss_sum += loss.data[0]
        loss.backward()
        optimizer.step()
    print ('epoch :',i, 'iterations :',iterr*batch_size, 'loss :',loss.data[0])
    torch.save(model.state_dict(), out_dir+'neural_4000.pth')

predicted = []
for iterr in range(num_batch):
    batch_data = Variable(torch.FloatTensor(test_data[iterr*batch_size:(iterr+1)*batch_size]).cuda())
    class_pred = model(batch_data)
    _, label = torch.max(class_pred.data, 1)
    for tt in range(len(label)):
        predicted.append(int(label[tt]))
out_name = "neural_2000.csv"
f = open(out_dir+out_name,"w")
f.write("ID,CATEGORY\n")
for i in range(len(predicted)):
    f.write(str(i)+","+file_labels[int(predicted[i])]+"\n")    

