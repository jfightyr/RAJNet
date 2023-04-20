import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from tqdm import tqdm
# import torch
# torch.distributed.init_process_group(backend="nccl") # 2022.09.07 多GPU初始化，采用此方法时需要在终端加入python -m torch.distributed.launch main.py
import math
from sklearn import metrics
from email import parser
from optparse import Option
from sklearn.metrics import recall_score, precision_score
import pandas as pd  # 对panda做处理
import torch.nn as nn  # 做深度学习！
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

# from dataset import BasicDataset, scipy_rotate
from dataset_h5f import BasicDataset
from metrics import *
from models import generate_model
from utils import *


# 数据预处理执行顺序：LabelMake4.1 -> ExcelProcess -> train
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
df = pd.read_csv('/home/huangjiehui/Project/HDAnaylis/ljy/Tools/Data.csv') 
# print(type(df)) <class 'pandas.core.frame.DataFrame'>
# pandas.DataFrame -- 二维矩阵
df_shuffle = df.sample(frac=1, random_state=100) # 抽样调查
# print(type(df_shuffle)) # <class 'pandas.core.frame.DataFrame'>
split_idx = int(len(df_shuffle) * 0.7) # 在原长0.7的位置切片 -- 因为机器学习一般7成训练3成测试
df_train = df_shuffle.iloc[:split_idx] 
df_test = df_shuffle.iloc[split_idx:]

# 路径和标题命个名？
train_data_path = list(df_train['data_path'])
train_labels = list(df_train['label_list'])

test_data_path = list(df_test['data_path'])
test_labels = list(df_test['label_list'])


train_transform = 1#  0927 改动scipy_rotate 不做旋转了 -- loss明显下降了  #11.18改动旋转
test_transform = None
#                             这两个让path和labels匹配，后面方便我们训练

from h5fMake import *
# 已做好0108
# prepareHDData(data_path = train_data_path,labels = train_labels,patch_size = 256 ,stride = 256,data_type='train')
# prepareHDData(data_path=test_data_path,labels=test_labels,patch_size=256,stride=256,data_type='test')
print("H5fMake Successful!")


train_set = BasicDataset(train_data_path, train_labels, data_type='train',transform=train_transform)
test_set = BasicDataset(test_data_path, test_labels, data_type='test',transform=test_transform)
                    # 数据抽样堆叠 -- 因为神经网络不能跑太久 
                    # 第三个参数 -- 是否打乱数据送到网络里面去 -- 这个数字越大
                    # num_worker就是一次调用1个线程
train_loader = DataLoader(train_set, batch_size=3, shuffle=True, num_workers=0) # 2022.08.16 10张V100，大胆尝试！Out
test_loader = DataLoader(test_set, batch_size=3, shuffle=False, num_workers=0) # 2022.09.11 batch_size 太小会使得loss很大
# 这里改了一下
"""
batch_size=64的意思是：一次随机抓取64张图片。
shuffle=True的意思是：抓完一轮以后洗牌。
num_workers=0的意思是：数据在主进程加载。
drop_last=True的意思是：丢弃最后一个不完整的抓取。
"""

# load model
# 模型实例化
# 模型深度固定为[10, 18, 34, 50, 101, 152, 200]中的一个
model = generate_model(model_depth=101, n_input_channels=1, n_classes=2).cuda() # 这里报错要么就是没有gpu的cuda~~~~~~~~_input_channels=M: you can work with Mx6x7 input arrays.
# 原本深度101
"""print(type(model)) # <class 'model.ResNet'>"""
#====Method 1
# model = nn.DataParallel(model) # 将模型对象转变为多GPU并行运算的模型 Jack 2022.09.07
# device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
# model.to(device)
#====Method 2
# local_rank = torch.distributed.get_rank() # 配置每个进程的gpu
# print('local_rank',local_rank)
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
# model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)
# model.cuda()  # 把并行的模型移动到GPU上，无需model.to(device)，DP方法会自动搜索device_id
# 参数设置
parser = argparse.ArgumentParser(description="JackNet_Train")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
opt = parser.parse_args()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-4) # Adam这个优化器比较重要
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # 变化学习率，很有效的一个参数


epochs = 100  
steps_per_epoch = 10
# num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
# num_iter=2

global_step = 0
global_epoch = 0

running_loss = AverageMeter()  # BUG loss和acc一样的
running_acc = AverageMeter()

model.train() # 激活这个模型
Loss = [1] # 初始化变量
Acc = [0]
TestLoss = [1]
TestAcc = [0]
Note=open('/home/huangjiehui/Project/HDAnaylis/ljy/Result/HD0108.txt','a') 
Note.truncate(0) # 初始化txt

initial_epoch = findLastCheckpoint(save_dir="//home/huangjiehui/Project/HDAnaylis/ljy/Result/logs")  # 存net地址
if initial_epoch > 0:
    print('resuming by loading epoch %d' % initial_epoch)
    # model = CTPnet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu).cuda()
    model.load_state_dict(torch.load(os.path.join("//home/huangjiehui/Project/HDAnaylis/ljy/Result/logs", 'net_epoch%d.pth' % initial_epoch)))


numsForAll=0
if __name__ == '__main__':
    # 训练才会被激活
    for i in range(1000):
        for data in tqdm(train_loader):

            global_step += 1

            image, label = data  # 后续加name

            image = image.cuda() # BUG CUDA out of memory
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
            # break

        if i+1%100:
            torch.save(model.state_dict(), os.path.join("/home/huangjiehui/Project/HDAnaylis/ljy/Result/logs", 'net_epoch%d.pth' % (i+1)))
            torch.save(model.state_dict(), os.path.join("/home/huangjiehui/Project/HDAnaylis/ljy/Result/logs", 'net_latest.pth'))

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

            Note=open('/home/huangjiehui/Project/HDAnaylis/ljy/Result/HD0402.txt','a')
            # Note.write("%d,%.4f,%.4f, %.4f, %.4f \r\n" % (global_epoch, epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc))
            Note.write("epoch: %d, trian_loss: %.4f, trian_acc: %.4f, test_loss: %.4f, test_acc: %.4f,recall:%.4f,precision:%.4f\n" % (global_epoch, epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc,epoch_recall,epoch_precision)  )      
    
        import matplotlib.pyplot as plt
        # 跑完1个epoch更新一次曲线图
        plt.figure(1)
        plt.plot(Loss, label='train loss')
        plt.plot(TestLoss, label='test loss')
        plt.title('Loss curve')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig("/home/huangjiehui/Project/HDAnaylis/ljy/Result/TestAccCurve_0402_roed.png")

        plt.figure(2)
        plt.plot(Acc, label='train accuracy')
        plt.plot(TestAcc, label='test accuracy')
        plt.title('Accuracy curve')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig("/home/huangjiehui/Project/HDAnaylis/ljy/Result/TrainAccCurve_0402_roed.png")

