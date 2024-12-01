import pandas as pd
import os
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision
import scipy
import copy
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import fnmatch
from skimage import io
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets, models
#from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, Subset, random_split,WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score , auc
from torch.utils.data.sampler import Sampler
#from tqdm import tqdm
from collections import defaultdict
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("set to device: ",device)

train_df = pd.read_csv("/workspace/data/kaggle-OV/all_patches/train.csv")
print(train_df)

classes = ['HGSC', 'LGSC', 'EC', 'CC', 'MC']
cdict = {c:idx for (idx, c) in enumerate(classes)}


inputdir = ['/workspace/data/kaggle-OV/all_patches']

label_dict = {}
idx_patch_dict = {}
uniq_label_ids = {}
for idir in inputdir:
    for path,dirs,files in os.walk(idir):
        for f in fnmatch.filter(files,'*.png'):
            idx = f.split('_')[0]
            path_base = os.path.basename(path)
            if path_base not in label_dict:
                uniq_label_ids[path_base] = [idx]
                label_dict[path_base]=[path + '/' + f]
            else:
                uniq_label_ids[path_base].append(idx)
                label_dict[path_base].append(path+'/'+f)
            
            if idx not in idx_patch_dict:
                idx_patch_dict[idx] = [path+'/'+f]
            else:
                idx_patch_dict[idx].append(path+'/'+f)

uniq_ids = []
for k,v in uniq_label_ids.items():
    for j in set(v):
        uniq_ids.append(int(j))
print(len(uniq_ids))
mod_train_df = train_df[train_df['image_id'].isin(uniq_ids)].reset_index()
print(mod_train_df)

train_idx, val_idx = train_test_split(mod_train_df.index, test_size=0.2, 
                                          shuffle=True, stratify=mod_train_df.label)
print(train_idx)
print(val_idx)

train_label_idx_dict = {}
for i in train_idx:
    row = mod_train_df.iloc[i]
    lab = row['label']
    imgid = str(row['image_id'])
    if lab in train_label_idx_dict:
        train_label_idx_dict[lab].append(imgid)
    else:
        train_label_idx_dict[lab] = [imgid]

#print(train_label_idx_dict)

val_label_idx_dict = {}
for i in val_idx:
    row = mod_train_df.iloc[i]
    lab = row['label']
    imgid = str(row['image_id'])
    if lab in val_label_idx_dict:
        val_label_idx_dict[lab].append(imgid)
    else:
        val_label_idx_dict[lab] = [imgid]

#print(val_label_idx_dict)



#print([(k,len(v)) for k,v in train_label_idx_dict.items()])
def get_label_patch(idx_dict,idx_patch_dict):
    final_lst_patch = []
    for label in idx_dict:
        for ids in idx_dict[label]:
            if ids in idx_patch_dict:
                for path in idx_patch_dict[ids]:
                    final_lst_patch.append((label,path))
    
    final_df  = pd.DataFrame(final_lst_patch, columns=['label','path_patch'])
    return final_df
#print(final_lst_patch)
final_train_df  = get_label_patch(train_label_idx_dict,idx_patch_dict)
final_val_df = get_label_patch(val_label_idx_dict,idx_patch_dict)
print(final_train_df)
print(final_val_df)

########################################################################################################################
class OCdataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        #label = cdict[self.df.loc[df.index[idx], 'label']]
        label = cdict[self.df['label'][idx]]
        #path = self.df.loc[df.index[idx], 'tile_path']
        path = self.df['path_patch'][idx]
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        #print(img,label)
        return img, label

#################################################################################################################################   
class OCValidDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        label = cdict[self.df['label'][idx]]
        path = self.df['path_patch'][idx]
        imgid = path.split('/')[-1].split('_')[0]
        
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        return img, label, imgid
    
#####################################################################################################################################
def buildDatasets(train_df,val_df):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    ## Add normalization and comment out the resnet18_weights
    
    train_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.1), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)
                                ])
    
    valid_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)
                                ])
    
    train_dataset = OCdataset(train_df, transform=train_transform)
    validation_dataset = OCValidDataset(val_df, transform=valid_transform)
    print(len(train_dataset))
    print(len(validation_dataset))
    return train_dataset, validation_dataset



###
#final_train_df = final_train_df.sample(frac=1).reset_index()[:5000]
#final_val_df = final_val_df.sample(frac=1).reset_index(drop=True)[:100]

get_train = buildDatasets(final_train_df, final_val_df)
train_dataset = get_train[0]
val_dataset = get_train[1]
##################################################################################
class labelsampler(Sampler):
    def __init__(self,dataset,pct=0.2):
        self.dataset = dataset
        self.n = len(dataset)
        self.pct = pct
        
    def __iter__(self):
        hgsc_idxs = np.where(self.dataset.df['label']=='HGSC')[0]
        lgsc_idxs = np.where(self.dataset.df['label']=='LGSC')[0]
        ec_idxs = np.where(self.dataset.df['label']=='EC')[0]
        cc_idxs = np.where(self.dataset.df['label']=='CC')[0]
        mc_idxs = np.where(self.dataset.df['label']=='MC')[0]
        #rest_idxs = np.where(-self.dataset.df.target)[0]
        #print(rest_idxs)
        hgsc = np.random.choice(hgsc_idxs, int(self.n * self.pct),replace = True)
        lgsc = np.random.choice(lgsc_idxs, int(self.n * self.pct),replace = True)
        ec = np.random.choice(ec_idxs, int(self.n * self.pct),replace = True)
        cc = np.random.choice(cc_idxs, int(self.n * self.pct),replace = True)
        mc = np.random.choice(mc_idxs, int(self.n * self.pct),replace = True)
        idxs = np.hstack([hgsc,lgsc,ec,cc,mc])
        np.random.shuffle(idxs)
        idxs = idxs[:self.n]
        return iter(idxs)
    def __len__(self):
        return self.n
    
##################################################################################
train_loader = DataLoader(train_dataset, batch_size=15, num_workers=4
                              , sampler=labelsampler(train_dataset,pct = 0.2))
val_loader = DataLoader(val_dataset, batch_size=15, num_workers=4, 
                            pin_memory=True, persistent_workers=True)
print(type(train_loader))


        
dataiter = iter(train_loader)
images = next(dataiter)
images[0].shape


print(len(train_loader))

##############################################################################################################################


def build_model(num_classes, model_type = "resnet50", freeze_convnets = True, single_layer = True, hidden_units = 32):
    #Load pre-trained model
    if model_type == "resnet18":
        model = models.resnet18(pretrained = True)
    elif model_type =="resnet50":
        model = models.resnet50(pretrained = True)
    elif model_type =="resnet101":
        model = models.resnet101(pretrained = True)
    elif model_type =="resnet152":
        model = models.resnet152(pretrained = True)
    else:
        model = models.resnet18(pretrained = True)
    
    
    
    #Get the output dimension from the Conv block
    num_ftrs = model.fc.in_features
    
    #Parameters of newly constructed modules have requires_grad = True by default
    #Here the size of each output sample is set to num_classes
    if single_layer:
        model.fc = nn.Linear(num_ftrs, num_classes)
        '''model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )'''
    else:
        model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, num_classees)
                )
    nn.init.xavier_uniform_(model.fc.weight)
    #Freeze the convnets
    if freeze_convnets:
        for param in model.parameters():
            param.requires_grad = True

    return model

##########################################################################################################################
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

########################################################################################################################


def train(net, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, optim='sgd', init=True, scheduler_type='Cosine'):
    '''def init_xavier(m): 
        #if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        net.apply(init_xavier)'''

    print('training on:', device)
    net.to(device)
    
    if optim == 'sgd': 
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=0)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=1e-3)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=0)
    elif optim == 'ranger':
        optimizer = Ranger((param for param in net.parameters() if param.requires_grad), lr=lr,
                           weight_decay=0)
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=7, eta_min=lr_min)
    
    train_losses = []
    train_acces = []
    eval_acces = []
    #eval_loss_list = []
    best_acc = 0.0
    
    min_eval_loss = 99999999
   
    
    early_stopper = EarlyStopper(patience=10, min_delta=0)
    #Train
    for epoch in range(num_epoch):

        print("Start of training round {}".format(epoch + 1))

        
        net.train()
        train_acc = 0
        for batch in train_dataloader:
            imgs, targets = batch
            imgs = imgs.to(device)
            #targets = torch.cat(targets, dim = 0)
            targets = targets.to(device)
            output = net(imgs)

            Loss = loss(output, targets)
          
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / (batch_size)
            train_acc += acc
        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch, Loss.item(), train_acc / len(train_dataloader)))
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(Loss.item())

        
        net.eval()
        eval_loss = 0
        eval_acc = 0
        y_valid = []
        y_pred = []
        y_imgids = []
        with torch.no_grad():
            for imgs, targets,imgids in valid_dataloader:
                imgs = imgs.to(device)
                #print('targets', targets[0])
                #print('ingids', imgids[0])

                #targets = torch.cat(targets, dim = 0)
                #y_valid += targets
                #y_imgids += imgids
                #y_valid.extend(targets.numpy())
                y_imgids.extend(imgids)
                #print(y_valid)
                #print(y_imgids)
                targets = targets.to(device)
                output = net(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                               
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss.item()
                #print("eval_loss: ", eval_loss)
                acc = num_correct / imgs.shape[0]
                eval_acc += acc
                y_pred += output.cpu().numpy().tolist()
                #print(sys.getsizeof(y_pred))
                #print(y_pred[0])
                #print(y_pred.shape)
            eval_losses = eval_loss / (len(valid_dataloader))
            if eval_losses < min_eval_loss:
                min_eval_loss = eval_losses
                checkpoint = {
                    'epoch' : epoch,
                    'model' : net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'lr_sched' : scheduler
                }
                torch.save(checkpoint, '/workspace/data/kaggle-OV/models/bilal/Jan1_best_chkpnt_resnet18_batch20-epoch100_pat10_tmax7_adam.pth')
                #torch.save(net.state_dict(),'/workspace/data/kaggle-OV/models/naina/resnet50_batch20-epoch100_pat10_tmax7_adam/best_chkpnt_resnet50_batch20-epoch100_pat10_tmax7_adam_1.pth')
                y_pred_final = copy.deepcopy(y_pred)
                y_img_id_final = copy.deepcopy(y_imgids)
                #y_valid_final = y_valid
            #eval_loss_list.append(eval_losses)
            del y_pred
            del y_imgids
            
            eval_acc = eval_acc / (len(valid_dataloader))
            #if eval_acc > best_acc:
            #    best_acc = eval_acc
            #    torch.save(net.state_dict(),'/kaggle/working/best_checkpoint_0_100.pth')
            eval_acces.append(eval_acc)
            
            print("Loss on the overall validation set: {}".format(eval_losses))
            print("Correctness on the overall validation set: {}".format(eval_acc))
            print('Last learning rate:',scheduler.get_last_lr())
            if early_stopper.early_stop(eval_losses):
                print('Done early stopping with epoch:',epoch)
                break
    #print(y_pred_final)
    #print(y_img_id_final)
    return train_losses, train_acces, eval_acces, y_pred_final, y_img_id_final
#########################################################################################################################################

def show_acces(train_losses, train_acces, eval_acces, num_epoch):
    plt.plot(1 + np.arange(len(train_losses)), train_losses, linewidth=1.5, linestyle='dashed', label='train_losses')
    plt.plot(1 + np.arange(len(train_acces)), train_acces, linewidth=1.5, linestyle='dashed', label='train_acces')
    plt.plot(1 + np.arange(len(eval_acces)), eval_acces, linewidth=1.5, linestyle='dashed', label='eval_acces')
    plt.grid()
    plt.xlabel('epoch')
    plt.xticks(range(1, 1 + num_epoch, 1))
    plt.legend()
    #plt.show()
    plt.savefig('/workspace/data/kaggle-OV/plots/bilal/Jan1_lr_0p03_minlr_0p0005_Resnet18_train_eval_loss_acc_batch20-epoch100_pat10_tmax7_adam.png')

#######################################################################################################################################

net = build_model(num_classes = 5, model_type = "resnet18", freeze_convnets = True, single_layer = True, hidden_units = 64)

print("Model:",net)
loss = nn.CrossEntropyLoss()
train_losses, train_acces, eval_acces, y_pred, y_imgids = train(net, loss, train_loader, val_loader, device, batch_size=15, num_epoch=100, lr=0.03, lr_min=0.0005, optim='adam', init=False)


show_acces(train_losses, train_acces, eval_acces, num_epoch=10)

#######################################################################################################################################

#y_valid1 = [x.numpy() for x in y_valid]
#y_valid1 = np.array(y_valid).flatten().tolist()
#print(y_valid1)

#y_imgids1 = [x.numpy() for x in y_imgids]
#y_imgids1 = np.array(y_imgids1).flatten().tolist()
#y_imgids1 = set(y_imgids)
#print(y_imgids1)


#image_wise_info_valid = []
y_valid_image_label = {}
for label in val_label_idx_dict:
    for ids in val_label_idx_dict[label]:
        if ids in idx_patch_dict:
            #num_of_patches = len(idx_patch_dict[ids])
            #image_wise_info_valid.append((ids,cdict[label],num_of_patches))
            y_valid_image_label[ids] = cdict[label]
#print(image_wise_info_valid)

#cdict_rev = {idx:c for (c,idx) in cdict.items()}
def get_label(predictions,method):
    probs = scipy.special.softmax(predictions, axis=1)
    if method == 'agg':
        probs_final = np.mean(probs, axis=0)
    elif method == 'logsum':
        probs = np.log(probs)
        probs_final = np.sum(probs,axis=0)
    label = np.argmax(probs_final)
    return label


        
        
        
id_wise_patch_labels = {}
for i in range(len(y_imgids)):
    if y_imgids[i] in id_wise_patch_labels:
        id_wise_patch_labels[y_imgids[i]].append(y_pred[i])
    else:
        id_wise_patch_labels[y_imgids[i]] = [y_pred[i]]

y_valid2 = []
y_pred2 = []
y_valid_pred_label = {}
for ids in id_wise_patch_labels:
    y_valid_pred_label[ids] = get_label(id_wise_patch_labels[ids],'agg')
    if ids in y_valid_image_label:

        y_valid2.append(y_valid_image_label[ids])
        y_pred2.append(y_valid_pred_label[ids])
    
print(y_valid2, y_pred2)

print('At image level')
print("Confusion Matrix:\n",confusion_matrix(y_valid2,y_pred2))
print()
print("Classification Report:\n",classification_report(y_valid2,y_pred2))



