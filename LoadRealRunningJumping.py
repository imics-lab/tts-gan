#A binary classification dataset, Jumping or Running


import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
#from tensorflow.keras.utils import to_categorical # for one-hot encoding

#credit https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
#many other methods I tried failed to download the file properly
from torch.utils.data import Dataset, DataLoader

class_dict = {'StandingUpFS':0,'StandingUpFL':1,'Walking':2,'Running':3,'GoingUpS':4,'Jumping':5,'GoingDownS':6,'LyingDownFS':7,'SittingDown':8}

class Running_Or_Jumping(Dataset):
    def __init__(self, 
        incl_xyz_accel = False, #include component accel_x/y/z in ____X data
        incl_rms_accel = True, #add rms value (total accel) of accel_x/y/z in ____X data
        is_normalize = False,
        split_subj = dict
                    (train_subj = [4,5,6,7,8,10,11,12,14,15,19,20,21,22,24,26,27,29,1,9,16,23,25,28],
                    test_subj = [2,3,13,17,18,30]),
        data_mode = 'Train'):
        
        self.incl_xyz_accel = incl_xyz_accel
        self.incl_rms_accel = incl_rms_accel
        self.split_subj = split_subj
        self.data_mode = data_mode
        self.is_normalize = is_normalize
        
        #Download and unzip original dataset
        if (not os.path.isfile('./UniMiB-SHAR.zip')):
            print("Downloading UniMiB-SHAR.zip file")
            #invoking the shell command fails when exported to .py file
            #redirect link https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            #!wget https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            self.download_url('https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip','./UniMiB-SHAR.zip')
        if (not os.path.isdir('./UniMiB-SHAR')):
            shutil.unpack_archive('./UniMiB-SHAR.zip','.','zip')
        #Convert .mat files to numpy ndarrays
        path_in = './UniMiB-SHAR/data'
        #loadmat loads matlab files as dictionary, keys: header, version, globals, data
        adl_data = io.loadmat(path_in + '/adl_data.mat')['adl_data']
        adl_names = io.loadmat(path_in + '/adl_names.mat', chars_as_strings=True)['adl_names']
        adl_labels = io.loadmat(path_in + '/adl_labels.mat')['adl_labels']

        #Reshape data and compute total (rms) acceleration
        num_samples = 151 
        #UniMiB SHAR has fixed size of 453 which is 151 accelX, 151 accely, 151 accelz
        adl_data = np.reshape(adl_data,(-1,num_samples,3), order='F') #uses Fortran order
        if (self.incl_rms_accel):
            rms_accel = np.sqrt((adl_data[:,:,0]**2) + (adl_data[:,:,1]**2) + (adl_data[:,:,2]**2))
            adl_data = np.dstack((adl_data,rms_accel))
        #remove component accel if needed
        if (not self.incl_xyz_accel):
            adl_data = np.delete(adl_data, [0,1,2], 2)
            
        #Split train/test sets, combine or make separate validation set
        #ref for this numpy gymnastics - find index of matching subject to sub_train/sub_test/sub_validate
        #https://numpy.org/doc/stable/reference/generated/numpy.isin.html


        act_num = (adl_labels[:,0])-1 #matlab source was 1 indexed, change to 0 indexed
        sub_num = (adl_labels[:,1]) #subject numbers are in column 1 of labels


        train_index = np.nonzero(np.isin(sub_num, self.split_subj['train_subj']))
        x_train = adl_data[train_index]
        y_train = act_num[train_index]

        test_index = np.nonzero(np.isin(sub_num, self.split_subj['test_subj']))
        x_test = adl_data[test_index]
        y_test = act_num[test_index]

        self.x_train = np.transpose(x_train, (0, 2, 1))
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1, self.x_train.shape[2])
        self.x_train = self.x_train[:,:,:,:-1]
        self.y_train = y_train
        
        self.x_test = np.transpose(x_test, (0, 2, 1))
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1, self.x_test.shape[2])
        self.x_test = self.x_test[:,:,:,:-1]
        self.y_test = y_test
        
        if self.is_normalize:
            self.x_train = self.normalization(self.x_train)
            self.x_test = self.normalization(self.x_test)
        
        #Select running and jumping data 
        #Label running as 0 and jumping as 1
        
        Jumping_train_data = []
        Running_train_data = []
        Jumping_test_data = []
        Running_test_data = []


        for i, label in enumerate(y_train):
            if label == class_dict['Running']:
                Running_train_data.append(self.x_train[i])
            elif label == class_dict['Jumping']:
                Jumping_train_data.append(self.x_train[i])
            else:
                continue
                
        for i, label in enumerate(y_test):
            if label == class_dict['Running']:
                Running_test_data.append(self.x_test[i])
            elif label == class_dict['Jumping']:
                Jumping_test_data.append(self.x_test[i])
            else:
                continue
                
        self.Jumping_train_labels = np.ones(len(Jumping_train_data))
        self.Jumping_test_labels = np.ones(len(Jumping_test_data))
        self.Running_train_labels = np.zeros(len(Running_train_data))
        self.Running_test_labels = np.zeros(len(Running_test_data))

        self.Jumping_train_data = np.array(Jumping_train_data)
        self.Running_train_data = np.array(Running_train_data)
        self.Jumping_test_data = np.array(Jumping_test_data)
        self.Running_test_data = np.array(Running_test_data)
        
        
        #Crop Running to only 600 samples
        self.Running_train_data = self.Running_train_data[:600][:][:][:]
        self.Running_train_labels = self.Running_train_labels[:600]
        
        self.Running_test_data = self.Running_test_data[:146][:][:][:]
        self.Running_test_labels = self.Running_test_labels[:146]
        
        self.combined_train_data = np.concatenate((self.Jumping_train_data, self.Running_train_data), axis=0)
        self.combined_test_data = np.concatenate((self.Jumping_test_data, self.Running_test_data), axis=0)
        
        self.combined_train_label = np.concatenate((self.Jumping_train_labels, self.Running_train_labels), axis=0)
        self.combined_train_label = self.combined_train_label.reshape(self.combined_train_label.shape[0], 1)
        
        self.combined_test_label = np.concatenate((self.Jumping_test_labels, self.Running_test_labels), axis=0)
        self.combined_test_label = self.combined_test_label.reshape(self.combined_test_label.shape[0], 1)
        
        if self.data_mode == 'Train':
            print(f'data shape is {self.combined_train_data.shape}, label shape is {self.combined_train_label.shape}')
            print(f'Jumping label is 1, has {len(self.Jumping_train_labels)} samples, Running label is 0, has {len(self.Running_train_labels)} samples')
        else:
            print(f'data shape is {self.combined_test_data.shape}, label shape is {self.combined_test_label.shape}')
            print(f'Jumping label is 1, has {len(self.Jumping_test_labels)} samples, Running label is 0, has {len(self.Running_test_labels)} samples')
        
        
    def download_url(self, url, save_path, chunk_size=128):
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]       

    
    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / ((np.sqrt(epoch.var(axis=0)))+e)
        return result
    
    def _min_max_normalize(self, epoch):
        
        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result

    def normalization(self, epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
            Args:
                epochs - Numpy structure of epochs
            Returns:
                epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                epochs[i,j,0,:] = self._normalize(epochs[i,j,0,:])
#                 epochs[i,j,0,:] = self._min_max_normalize(epochs[i,j,0,:])

        return epochs
    
    
    def __len__(self):
        
        if self.data_mode == 'Train':
            return len(self.combined_train_label)
        else:
            return len(self.combined_test_label)
        
    def __getitem__(self, idx):
        
        if self.data_mode == 'Train':
            return self.combined_train_data[idx], self.combined_train_label[idx]
        else:
            return self.combined_test_data[idx], self.combined_test_label[idx]
    
    def collate_fn(self):
        pass
        
            