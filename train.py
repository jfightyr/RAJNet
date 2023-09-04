import os
import argparse
from tqdm import tqdm
import math
from sklearn import metrics
from email import parser
from optparse import Option
from sklearn.metrics import recall_score, precision_score
import pandas as pd 
import torch.nn as nn  
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset


from dataset_h5f import BasicDataset
from metrics import *
from models import generate_model
from utils import *


def eval(model, eval_loader, criterion):
    model.eval()

    running_loss = AverageMeter()
    running_acc = AverageMeter()
    y_true=y_pred=[]
    running_precision=0.0
    running_recall=0.0
    for data in tqdm(eval_loader):
        image, label = data
        image = image.cuda()
        label = label.cuda()

        with torch.no_grad():
            logit = model(image)
            loss = criterion(logit, label)
            pred = torch.argmax(logit, dim=-1)
            acc = accuracy(label.cpu().numpy(), pred.cpu().numpy())  
            y_true.append(label)
            y_pred.append(pred) 
            

        running_loss.update(loss.item(), image.size(0))
        running_acc.update(acc, image.size(0))
        running_precision+=metrics.precision_score(label.data.cpu().numpy(), pred.cpu().numpy())  #精确率
        running_recall+=metrics.recall_score(label.data.cpu().numpy(), pred.cpu().numpy())   #召回率
        # break

    epoch_loss = running_loss.get_average()
    epoch_acc = running_acc.get_average()
    epoch_recall = running_recall / len(eval_loader)
    epoch_precision = running_precision / len(eval_loader)

    running_loss.reset()
    running_acc.reset()

    model.train()

    return epoch_loss, epoch_acc,epoch_recall,epoch_precision



# load data 
df = pd.read_csv('./Tools/Data.csv') 
df_shuffle = df.sample(frac=1, random_state=100) 
split_idx = int(len(df_shuffle) * 0.7) 
df_train = df_shuffle.iloc[:split_idx] 
df_test = df_shuffle.iloc[split_idx:]

train_data_path = list(df_train['data_path'])
train_labels = list(df_train['label_list'])

test_data_path = list(df_test['data_path'])
test_labels = list(df_test['label_list'])


train_transform = 1
test_transform = None

from h5fMake import *

print("H5fMake Successful!")


train_set = BasicDataset(train_data_path, train_labels, data_type='train',transform=train_transform)
test_set = BasicDataset(test_data_path, test_labels, data_type='test',transform=test_transform)

train_loader = DataLoader(train_set, batch_size=3, shuffle=True, num_workers=0) 
test_loader = DataLoader(test_set, batch_size=3, shuffle=False, num_workers=0) 



model = generate_model(model_depth=101, n_input_channels=1, n_classes=2).cuda() 
parser = argparse.ArgumentParser(description="JackNet_Train")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
opt = parser.parse_args()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)

epochs = 100  
steps_per_epoch = 10
num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))

global_step = 0
global_epoch = 0

running_loss = AverageMeter() 
running_acc = AverageMeter()

model.train() 
Loss = [1]
Acc = [0]
TestLoss = [1]
TestAcc = [0]
Note=open('./Result/HD0108.txt','a') 
Note.truncate(0)

initial_epoch = findLastCheckpoint(save_dir="/./Result/logs")
if initial_epoch > 0:
    print('resuming by loading epoch %d' % initial_epoch)
    model.load_state_dict(torch.load(os.path.join("/./Result/logs", 'net_epoch%d.pth' % initial_epoch)))


numsForAll=0
if __name__ == '__main__':

    for i in range(1000):
        for data in tqdm(train_loader):

            global_step += 1

            image, label = data 

            image = image.cuda() 
            label = label.cuda()

            logit = model(image)

            loss = criterion(logit, label)
            pred = torch.argmax(logit, dim=-1)
            acc = accuracy(label.cpu().numpy(), pred.cpu().numpy())    

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), image.size(0))
            running_acc.update(acc, image.size(0))
            print("Epoch ",i)

        if i+1%100:
            torch.save(model.state_dict(), os.path.join("./Result/logs", 'net_epoch%d.pth' % (i+1)))
            torch.save(model.state_dict(), os.path.join("./Result/logs", 'net_latest.pth'))

        test_precision=0.0
        test_recall=0.0

        if global_step:

            global_epoch += 1

            epoch_loss = running_loss.get_average()
            epoch_acc = running_acc.get_average()

            running_loss.reset()
            running_acc.reset()

            epoch_test_loss, epoch_test_acc,epoch_recall ,epoch_precision= eval(model, test_loader, criterion)
            msg ="epoch: %d, loss: %.4f, acc: %.4f, test_loss: %.4f, test_acc: %.4f,recall:%.4f,precision:%.4f\n" % (global_epoch, epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc,epoch_recall,epoch_precision)
        
            Loss.append(epoch_loss)
            Acc.append(epoch_acc)
            TestLoss.append(epoch_test_loss)
            TestAcc.append(epoch_test_acc)
            print(msg)

            Note=open('./Result/HD0402.txt','a')
            Note.write("epoch: %d, trian_loss: %.4f, trian_acc: %.4f, test_loss: %.4f, test_acc: %.4f,recall:%.4f,precision:%.4f\n" % (global_epoch, epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc,epoch_recall,epoch_precision)  )      
    
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(Loss, label='train loss')
        plt.plot(TestLoss, label='test loss')
        plt.title('Loss curve')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig("./Result/TestAccCurve_0402_roed.png")

        plt.figure(2)
        plt.plot(Acc, label='train accuracy')
        plt.plot(TestAcc, label='test accuracy')
        plt.title('Accuracy curve')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig("./Result/TrainAccCurve_0402_roed.png")

