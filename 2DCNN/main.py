import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import os
from torch import optim
import pandas as pd
import numpy as np
from h5fMake_2DCNN import prepareHDData

from dataset_h5f_2DCNN import BasicDataset, scipy_rotate  

df = pd.read_csv('./Tools/Data.csv')
df_shuffle = df.sample(frac=1, random_state=100) 
split_idx = int(len(df_shuffle) * 0.7)
df_train = df_shuffle.iloc[:split_idx] 
df_test = df_shuffle.iloc[split_idx:]
train_data_path = list(df_train['data_path'])
train_labels = list(df_train['label_list'])

test_data_path = list(df_test['data_path'])
test_labels = list(df_test['label_list'])
train_set = BasicDataset(train_data_path, train_labels,data_type='train')  
test_set = BasicDataset(test_data_path, test_labels,data_type='test')
print("data successfully!")

train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0) 
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
 

class LeNet(nn.Module):      
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

        self.fc1 = nn.Linear(65536, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):       
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      
        x = x.view(x.size()[0], -1)     
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)     

 
net = LeNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
net = LeNet().to(device)   
 
 

loss_fuc = nn.CrossEntropyLoss()   
optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum = 0.9)  
 

EPOCH = 50
for epoch in range(EPOCH):
    sum_loss = 0
    right=sumnum=0
    for i, data in enumerate(train_loader):
        inputs, labels = data  
        inputs, labels = inputs.to(device), labels.to(device)
        image=inputs.cpu().numpy() 
        image=np.squeeze(image,axis=0) 
        image=np.transpose(image,(1,0,2,3)) 
        inputs = torch.tensor(image)
        device = torch.device('cuda:0')
        inputs = inputs.to(device)

        loss_sum=0
        all=0
        class0=class1=0

        for new_i in inputs:
            new_i=new_i[np.newaxis]           
            output = net(new_i)
            if output[0][0]<output[0][1]: 
                class1+=1
            else:
                class0+=1
            loss = loss_fuc(output, labels) 
            print('Single loss:',loss)
            loss_sum+=loss
            all+=1
        if class0>class1:
            pred=0
        else:
            pred=1
        if pred==labels:
            right+=1
        sumnum+=1
        optimizer.zero_grad()

        loss_sum.backward()  
        optimizer.step()
        print('train_loss:',loss_sum)

        sum_loss += loss_sum.item()  
        if i % 500 == 499:
            print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, sum_loss / 500))
            loss_backup=sum_loss / 500
            sum_loss = 0.0
        if i==10:
            break
    train_acc=right/sumnum

    correct = 0
    total = 0
    for data in test_loader:
        test_inputs, labels = data
        test_inputs, labels = test_inputs.to(device), labels.to(device)
        image=test_inputs.cpu().numpy() 
        image=np.squeeze(image,axis=0)
        image=np.transpose(image,(1,0,2,3)) 
        test_inputs = torch.tensor(image)
        device = torch.device('cuda:0')
        test_inputs = test_inputs.to(device)
        outputs_test = net(test_inputs)

        class0=class1=0
        for obj in outputs_test:
            if obj[0]>obj[1]:
                class0+=1
            else:
                class1+=1
        if class0>class1:
           predicted=0
        else: 
           predicted=1       

        total += labels.size(0)  
        correct += (predicted == labels).sum() 
        if total==10:
            break
        
    print('The accuracy of Epoch {}: {}'.format(epoch + 1, correct.item() / total))
    Note=open('./2DCNN/2DCNN0313.txt','a')



