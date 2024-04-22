import glob
import numpy as np
import torch
import torch.utils.data as data_utils
import os
import scipy
import cv2
from utils import contours, extract_number, fft

def _norm_v_data(v_data):
    # # norm to range [0, 1] by row
    # min_r_v = np.min(v_data, axis=1)
    # max_r_v = np.max(v_data, axis=1)
    # numerator = np.subtract(v_data, min_r_v[:, np.newaxis])
    # denominator = (max_r_v - min_r_v)[:, np.newaxis]
    # v_data = np.where(denominator == 0, 0, np.divide(numerator, denominator))
    
    # # norm to range [0, 1] by col
    # min_c_v = np.min(v_data, axis=0)
    # max_c_v = np.max(v_data, axis=0)
    # numerator = np.subtract(v_data, min_c_v)
    # denominator = max_c_v - min_c_v
    # v_data = np.where(denominator == 0, 0, np.divide(numerator, denominator))
    
    # globally norm
    v_data = cv2.normalize(v_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # # globally norm after log
    # v_data = np.log(v_data + 1)
    # v_data = cv2.normalize(v_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # # maxscale
    # v_data = v_data / np.max(v_data)
    
    return v_data

def _std_seis_data(seis_data):
    # zscore
    seis_data = (seis_data - seis_data.mean()) / max(seis_data.std(), 1e-8)
    return seis_data

def _norm_seis_data(seis_data):
    # globally norm
    seis_data = cv2.normalize(seis_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return seis_data

def batch_read(config, training=True):    
    dataset_path = os.path.join(config.DATASET_PATH, \
        'train_data' if training else 'test_data')
    
    in_channels = config.IN_CHANNELS
    in_dim = config.IN_DSP_DIM
    out_channels = config.OUT_CHANNELS
    out_dim = config.OUT_DSP_DIM
    
    seis_dataset = np.empty((0, in_channels, *in_dim))
    v_dataset = np.empty((0, out_channels, *out_dim))
    c_dataset = np.empty((0, out_channels, *out_dim))
    
    seis_flist = list(glob.glob(dataset_path + '/seismic/seismic' + '*.mat')) \
        + list(glob.glob(dataset_path + '/seismic/seismic' + '*.npy'))
    v_flist = list(glob.glob(dataset_path + '/vmodel/vmodel' + '*.mat')) \
        + list(glob.glob(dataset_path + '/vmodel/vmodel' + '*.npy'))
    
    seis_flist.sort(key=extract_number)
    v_flist.sort(key=extract_number)
    
    for seis_f, v_f in zip(seis_flist, v_flist):
        if seis_f.endswith('.mat'):
            seis_data = scipy.io.loadmat(seis_f)['data']
            
            # for SEG dataset
            # (400, 301, 29) -> (29, 400, 301)
            seis_data = np.transpose(seis_data, (2, 0, 1))
            
            # # fft & zscore standardize seis_data (SEG only)
            # seis_data = fft(seis_data)
            # seis_data = _std_seis_data(seis_data)
            
            # (29, 400, 301) -> (1, 29, 400, 301)
            seis_data = seis_data[np.newaxis, :]
            
            v_data = scipy.io.loadmat(v_f)['data']
            v_data = _norm_v_data(v_data)
            c_data = contours(v_data).astype(np.float64)
            
            v_data = v_data[np.newaxis, np.newaxis, :]
            c_data = c_data[np.newaxis, np.newaxis, :]
            
        elif seis_f.endswith('.npy'):
            seis_data = np.load(seis_f)
                        
            for d0 in range(seis_data.shape[0]):
                for d1 in range(seis_data.shape[1]):
                    seis_data[d0, d1] = _norm_seis_data(seis_data[d0, d1])
            
            v_data = np.load(v_f).astype(np.float64) # (500, 1, 70, 70)
            c_data = v_data.copy()
            for d0 in range(v_data.shape[0]):
                v_data[d0, 0] = _norm_v_data(v_data[d0, 0])
                c_data[d0, 0] = contours(v_data[d0, 0]).astype(np.float64)
    
        seis_dataset = np.append(seis_dataset, seis_data, axis=0)
        v_dataset = np.append(v_dataset, v_data, axis=0)
        c_dataset = np.append(c_dataset, c_data, axis=0)
        
    return seis_dataset, v_dataset, c_dataset

def get_DataLoader(seis_dataset, v_dataset, c_dataset, batch_size, shuffle=True):
    torch_dataset = data_utils.TensorDataset(
        torch.from_numpy(seis_dataset).float(),
        torch.from_numpy(v_dataset).float(),
        torch.from_numpy(c_dataset).float()
        )
    loader = data_utils.DataLoader(torch_dataset,
                                   batch_size=batch_size,
                                   pin_memory=True,
                                   drop_last=True,
                                   shuffle=shuffle)
    return loader