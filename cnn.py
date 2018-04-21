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

kernel_siz = [1,3,5,7,9,11]
hidden_dim = [200,400,600,800,1000,1200]





strid = 1


class CNN(torch.nn.Module) :
    def __init__(self,features,hidden_units,output_layer,out_channels,kernel_siz,strid,pad):
        super(CNN,self).__init__()
        self.layer = torch.nn.Sequential(
        torch.nn.Conv2d(1,out_channels,kernel_size=kernel_siz,stride=strid,padding=pad),
        torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.inp_hidden_layer = out_channels * features // 4
        self.hidden = nn.Linear(self.inp_hidden_layer,hidden_units)
        self.predict = nn.Linear(hidden_units,output_layer)
        # self.hidden_dim = hidden_units
        # self. = num_layers
        # self.lstm = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        # self.linearOut = nn.Linear(2*hidden_dim,num_classes)
    def forward(self,inputs):
        inputs = inputs.view(-1,1,28,28)
        out = self.layer(inputs)
        out = out.view(-1,self.inp_hidden_layer)
        lin_hidden_out = self.hidden(out)
        nl_hidden_out = F.relu(lin_hidden_out)
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

data_label_pair = []
for i in range(len(train_data)):
    data_label_pair.append((actual_label[i],train_data[i]))
random.shuffle(data_label_pair)
data_label_pair_1 = data_label_pair[0:50000]
data_label_pair_2 = data_label_pair[50000:100000]


# for dim in hidden_dim:
#     for ker_size in kernel_siz:
dim = 128
ker_size = 3
pad = (28 - (28 - ker_size + 1))//2
cnn_layers = 128
model = CNN(num_feat,dim,num_clus,cnn_layers,ker_size,strid,pad)
model = model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 300
batch_size = 2000
num_batch = len(train_data) // batch_size
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
    print ('epoch :',i, 'iterations :',dim,'kernel_size',ker_size, 'loss :',loss.data[0])
    if loss.data[0] < 0.3:
        break
    torch.save(model.state_dict(), out_dir+'neural_big'+ str(dim) +'_' + str(ker_size) + '.pth')
predicted = []
for iterr in range(num_batch):
    batch_data = Variable(torch.FloatTensor(test_data[iterr*batch_size:(iterr+1)*batch_size]).cuda())
    class_pred = model(batch_data)
    _, label = torch.max(class_pred.data, 1)
    for tt in range(len(label)):
        predicted.append(int(label[tt]))
out_name = "neural_big" + str(dim) +'_' + str(ker_size) + "_" + str(cnn_layers) + ".csv"
f = open(out_dir+out_name,"w")
f.write("ID,CATEGORY\n")
for i in range(len(predicted)):
    f.write(str(i)+","+file_labels[int(predicted[i])]+"\n")    

