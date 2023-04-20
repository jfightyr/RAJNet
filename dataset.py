# %%
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
from scipy import ndimage
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def scipy_rotate(volume):
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False)
    volume[volume < 0] = 0 # Jack BUG 0618
    volume[volume > 1] = 1
    return volume



def py_cpu_nms(dets, thresh):

    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = y1 + dets[:, 3]
    x2 = x1 + dets[:, 2]
    
    scores = dets[:, 4]  # bbox打分
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    
    # keep为最后保留的边框
    keep = []
    
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]
        
    return keep

def calculate(bound, mask):  # 不需要去除重叠的框框

    x, y, w, h = bound

    area = mask[y:y+h, x:x+w]

    pos = area > 0 + 0

    score = np.sum(pos)/(w*h)

    return score

def nms_cnts(cnts, mask, min_area):

    nms_threshold=0.3  # 改吗

    bounds = [cv2.boundingRect(
        c) for c in cnts if cv2.contourArea(c) > min_area]  # 筛掉面积过小的轮廓框 返回

    if len(bounds) == 0:
        return []

    scores = [calculate(b, mask) for b in bounds]  # 得分？ 只保留得分最大的

    bounds = np.array(bounds)

    scores = np.expand_dims(np.array(scores), axis=-1)

    keep = py_cpu_nms(np.hstack([bounds, scores]), nms_threshold)

    return bounds[keep]

def pop(l, value):

    l.pop(0)
    l.append(value)

    return l

def frame_judge(IMG):

    NEW_IMG=[]
    # NEW_IMG=np.array(NEW_IMG)
    previous=[]

    for i in range(np.shape(IMG)[2]):
        NEW_IMG.append(IMG[:,:,i])

        
        threshold=50
        k_size=7
        iterations=3
        min_area=800
        es = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        # 二帧差法等？
        previous.append(IMG[:,:,i])  # previous现存灰度图
        raw = IMG[:,:,i].copy()
        temp = cv2.absdiff(IMG[:,:,i], previous[0]) # 计算绝对值差   gray:array   Qimage[0]?
        temp = temp.astype(np.uint8)

        #if i!=0:  # 第一张不要
        temp = cv2.medianBlur(temp, k_size)  # 中值滤波 bug:改uint

        ret, mask = cv2.threshold(
            temp, threshold, 255, cv2.THRESH_BINARY)  # 二值化

        mask = cv2.dilate(mask, es, iterations)  # 膨胀
        mask = cv2.erode(mask, es, iterations)  # 腐蚀

        # _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        #cnts为list结构，列表中每个元素代表一个边沿信息           

        bounds = nms_cnts(cnts, mask, min_area)

        if len(bounds)>=4:  # 矩形框个数大于等于4-130多  # 6:26  # 5:34
            print("New IMG Add!")
            from PIL import Image
            im = Image.fromarray(IMG[:,:,i])
            if im.mode == "F":
                im = im.convert('RGB')
            im.save("/home/huangjiehui/Project/HDAnaylis/ljy/AddedIMG/addIMG_%s.jpeg" % str(i))
            NEW_IMG.append(IMG[:,:,i])

               
        previous = pop(previous, raw)
        

    NEW_IMG=np.array(NEW_IMG)
    NEW_IMG=NEW_IMG.transpose(2,1,0)
    #transpose!

    return NEW_IMG

class BasicDataset(Dataset):
    #外部
    def __init__(self, image_path, label, transform=None):
        self.image_path = image_path
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        if idx==0:
            print()
        image = np.load(self.image_path[idx])
        lbl = self.label[idx]
        
        image = np.transpose(image, (2, 1, 0)) # (512, 512, 20)  调换数组的行列值的索引值，类似于求矩阵的转置
        #transpose原来是012 后面变成210 xyz变成zyx  也可以理解为旋转 
        #目的：使得数据多样化拓展（把斜变 切割等） 解决过拟合
        if self.transform: # 11.18 旋转图像 原本是倒着的 
            # image = self.transform(image) 
            image=np.rot90(image,-1)
        # ljy: 数据增强
        
        # image_new = frame_judge(image) # image_new是增强之后的，由原来的90张，扩展为现在的130张  1225：屏蔽
        # image = image_new  
        image = np.transpose(image,(2,1,0))
        
        image = image.astype(np.float32)
        # image = image[:,:,:20] # 10.28 20帧重新测试新数据
        image = np.expand_dims(image, axis=0) # Jack处理图像时即扩展了Z轴 (1, 20, 512, 20)


# %%
if __name__ == '__main__':
    df = pd.read_csv('Data/HDdata_processed.csv')  # 改
    df_shuffle = df.sample(frac=1, random_state=100)
    split_idx = int(len(df_shuffle) * 0.7)
    df_train = df_shuffle.iloc[:split_idx]
    df_test = df_shuffle.iloc[split_idx:]

    train_data_path = list(df_train['data_path'])
    train_labels = list(df_train['label_list'])

    test_data_path = list(df_test['data_path'])
    test_labels = list(df_test['label_list'])

    train_transform = scipy_rotate
    train_set = BasicDataset(train_data_path, train_labels, train_transform)

    image = train_set[0]
    # plt.imshow(image[30, :, :]) # image[0]====torch.Size([1, 512, 512, 331]) BUG



# import pandas as pd
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms.functional as TF
# import matplotlib.pyplot as plt
# from augment import *


# class BasicDataSet(Dataset):
#     def __init__(self, data_path, labels, transform=None):
#         self.data_path = data_path
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_path)

#     def __getitem__(self, idx):
#         image = np.load(self.data_path[idx])
#         label = self.labels[idx]
#         image = np.transpose(image, (2, 1, 0))

#         if self.transform:
#             image = self.transform(image)
        
#         image = image.copy()
#         image = np.expand_dims(image, axis=0)

#         return torch.from_numpy(image).float(), torch.tensor(label, dtype=torch.long)



# if __name__ == '__main__':
#     df = pd.read_csv('data.csv')
#     data_path = list(df['data_path'])
#     labels = list(df['label_list'])

#     train_transform = Transformer()
#     train_set = BasicDataSet(data_path, labels, train_transform)

#     image, label = train_set[0]
#     print(image.shape)
#     image_npy = image.numpy()[0].transpose(2, 1, 0)
#     print(image_npy.shape)

#     idx=30
#     plt.subplot(311)
#     plt.imshow(image_npy[:, :, idx+1])
#     plt.subplot(312)
#     plt.imshow(image_npy[:, :, idx+5])
#     plt.subplot(313)
#     plt.imshow(image_npy[:, :, idx+8])



# # %%
