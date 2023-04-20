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
    angles = [-20, -10, -5, 5, 10, 20]
    angle = random.choice(angles)
    volume = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False)
    volume[volume < 0] = 0 
    volume[volume > 1] = 1
    return volume



def py_cpu_nms(dets, thresh):

    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = y1 + dets[:, 3]
    x2 = x1 + dets[:, 2]
    
    scores = dets[:, 4]  # bbox打分
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:

        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= thresh)[0]
        
        order = order[inds + 1]
        
    return keep

def calculate(bound, mask): 

    x, y, w, h = bound

    area = mask[y:y+h, x:x+w]

    pos = area > 0 + 0

    score = np.sum(pos)/(w*h)

    return score

def nms_cnts(cnts, mask, min_area):

    nms_threshold=0.3 

    bounds = [cv2.boundingRect(
        c) for c in cnts if cv2.contourArea(c) > min_area] 

    if len(bounds) == 0:
        return []

    scores = [calculate(b, mask) for b in bounds]  

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

    previous=[]

    for i in range(np.shape(IMG)[2]):
        NEW_IMG.append(IMG[:,:,i])

        
        threshold=50
        k_size=7
        iterations=3
        min_area=800
        es = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        previous.append(IMG[:,:,i]) 
        raw = IMG[:,:,i].copy()
        temp = cv2.absdiff(IMG[:,:,i], previous[0]) 
        temp = temp.astype(np.uint8)

        temp = cv2.medianBlur(temp, k_size) 

        ret, mask = cv2.threshold(
            temp, threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.dilate(mask, es, iterations) 
        mask = cv2.erode(mask, es, iterations) 

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        

        bounds = nms_cnts(cnts, mask, min_area)

        if len(bounds)>=4:  
            print("New IMG Add!")
            from PIL import Image
            im = Image.fromarray(IMG[:,:,i])
            if im.mode == "F":
                im = im.convert('RGB')
            im.save("./AddedIMG/addIMG_%s.jpeg" % str(i))
            NEW_IMG.append(IMG[:,:,i])

               
        previous = pop(previous, raw)
        

    NEW_IMG=np.array(NEW_IMG)
    NEW_IMG=NEW_IMG.transpose(2,1,0)

    return NEW_IMG

class BasicDataset(Dataset):

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
        
        image = np.transpose(image, (2, 1, 0))

        if self.transform: 
            image=np.rot90(image,-1)

        image = np.transpose(image,(2,1,0))
        
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)


# %%
if __name__ == '__main__':
    df = pd.read_csv('Data/HDdata_processed.csv')
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


