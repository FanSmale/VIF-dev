from data_reader import batch_read
from utils import v_show
from config import Config
from models import GenWrapper, GANWrapper
from data_reader import get_DataLoader
import time
from loss import Metrics_short, Metrics_full
import torch
import os
import numpy as np

import warnings
# Disable all warnings
warnings.filterwarnings('ignore')

class VBF():
    def __init__(self, config):
        self.config = config
        
        self.SToV_model = GenWrapper(config, 'NestedUNet_SToV', 1)
        self.SToC_model = GenWrapper(config, 'NestedUNet_SToC', 1)
        
        self.VCToV_model = GenWrapper(config, 'EMD_VCToV', 2)
            
        self.device = self.SToV_model.device
        
        self.result_path = self.config.RESULT_PATH
        
        self.metrics_s = Metrics_short()
        self.metrics_f = Metrics_full()
        
        self.last_saved_epoch = self.config.MAX_EPOCHS // self.config.SAVE_EVERY_EPOCHS * self.config.SAVE_EVERY_EPOCHS - 1
        
    def load(self, epoch=None):
        if epoch is None:
            epoch=self.last_saved_epoch
            
        if self.MODE == 1:
            if self.config.MODEL_TYPE == 4:
                self.SToV_model.load(epoch)
                self.SToC_model.load(epoch)
        elif self.MODE == 2:
            if self.config.MODEL_TYPE == 1:
                self.SToV_model.load(epoch)
                
            elif self.config.MODEL_TYPE == 2:
                self.SToC_model.load(epoch)
            
            elif self.config.MODEL_TYPE == 3:
                self.SToV_model.load(epoch)
                self.SToC_model.load(epoch)
                
            else:
                self.SToV_model.load(epoch)
                self.SToC_model.load(epoch)
                self.VCToV_model.load(epoch)
        
    def save(self, epoch=None):
        if epoch is None:
            epoch=self.last_saved_epoch
            
        if self.MODE == 1: # only train mode is allowed to call this func
            if self.config.MODEL_TYPE == 1:
                self.SToV_model.save(epoch)
                
            elif self.config.MODEL_TYPE == 2:
                self.SToC_model.save(epoch)
            
            elif self.config.MODEL_TYPE == 3:
                self.SToV_model.save(epoch)
                self.SToC_model.save(epoch)
                
            elif self.config.MODEL_TYPE == 4:
                self.VCToV_model.save(epoch)
                
            elif self.config.MODEL_TYPE == 5:
                self.SToV_model.save(epoch)
                self.SToC_model.save(epoch)
                self.VCToV_model.save(epoch)
                
    def lr_scheduler_step(self, epoch=None):
        if self.config.MODEL_TYPE == 1:
            self.SToV_model.gen_lr_scheduler.step()
            
            if hasattr(self.SToV_model, 'dis_lr_scheduler'):
                self.SToV_model.dis_lr_scheduler.step()
            
        elif self.config.MODEL_TYPE == 2:
            self.SToC_model.gen_lr_scheduler.step()
            
            if hasattr(self.SToC_model, 'dis_lr_scheduler'):
                self.SToC_model.dis_lr_scheduler.step()
        
        elif self.config.MODEL_TYPE == 3:
            self.SToV_model.gen_lr_scheduler.step()
            self.SToC_model.gen_lr_scheduler.step()
            
            if hasattr(self.SToV_model, 'dis_lr_scheduler'):
                self.SToV_model.dis_lr_scheduler.step()
            if hasattr(self.SToC_model, 'dis_lr_scheduler'):
                self.SToC_model.dis_lr_scheduler.step()
            
        elif self.config.MODEL_TYPE == 4:
            self.VCToV_model.gen_lr_scheduler.step()
            
            if hasattr(self.VCToV_model, 'dis_lr_scheduler'):
                self.VCToV_model.dis_lr_scheduler.step()
            
        elif self.config.MODEL_TYPE == 5:
            self.SToV_model.gen_lr_scheduler.step()
            self.SToC_model.gen_lr_scheduler.step()
            self.VCToV_model.gen_lr_scheduler.step()
            
            if hasattr(self.SToV_model, 'dis_lr_scheduler'):
                self.SToV_model.dis_lr_scheduler.step()
            if hasattr(self.SToC_model, 'dis_lr_scheduler'):
                self.SToC_model.dis_lr_scheduler.step()
            if hasattr(self.VCToV_model, 'dis_lr_scheduler'):
                self.VCToV_model.dis_lr_scheduler.step()
        
    def train(self):
        print('\n#################')
        print('#  Train Phase  #')
        print('#################\n')
        
        self.MODE = 1 # train mode
        
        # Get data
        seis_dataset, v_dataset, c_dataset = batch_read(config, self.MODE == 1)
        
        loader = get_DataLoader(seis_dataset, v_dataset, c_dataset, config.BATCH_SIZE)
        
        # Load models
        self.load()
        
        num_batches = len(loader)
        
        max_epochs = self.config.MAX_EPOCHS
        
        # Activate the train mode!
        self.SToV_model.train()
        self.SToC_model.train()
        self.VCToV_model.train()
        
        model_type = self.config.MODEL_TYPE
        device = self.device
        save_every_epochs = self.config.SAVE_EVERY_EPOCHS
        use_rand_crop_trans = True if self.config.USE_RAND_CROP_TRANS == 1 else False
        crop_ratio = self.config.CROP_RATIO
        
        epoch_loss_recorder = np.zeros(max_epochs)
        
        training_time = 0
        
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            cur_time = time.time()
            for items in loader:
                
                seis_s, v_s, c_s = items
                
                seis_s = seis_s.to(device)
                v_s = v_s.to(device)
                c_s = c_s.to(device)
                
                # SToV
                if model_type == 1:
                    SToV_outputs, SToV_loss_arr = self.SToV_model.process(seis_s, v_s)
                    
                    # cc metrics
                    v_loss = self.metrics_s(SToV_outputs, v_s)
                    # print('Loss v_loss: {:.12f}'.format(v_loss))
                    loss = v_loss
                    
                    self.SToV_model.backward(SToV_loss_arr)
                    
                # SToC
                if model_type == 2:
                    SToC_outputs, SToC_loss_arr = self.SToC_model.process(seis_s, c_s)
                    
                    # cc metrics
                    c_loss = self.metrics_s(SToC_outputs, c_s)
                    # print('Loss c_loss: {:.12f}'.format(c_loss))
                    loss = c_loss
                    
                    self.SToC_model.backward(SToC_loss_arr)
                
                # SToV & SToC
                elif model_type == 3:
                    SToV_outputs, SToV_loss_arr = self.SToV_model.process(seis_s, v_s)
                    SToC_outputs, SToC_loss_arr = self.SToC_model.process(seis_s, c_s)
                    
                    # cc metrics
                    v_loss = self.metrics_s(SToV_outputs, v_s)
                    c_loss = self.metrics_s(SToC_outputs, c_s)
                    # print('Loss v_loss: {:.12f}, c_loss: {:.12f}'\
                    #     .format(v_loss, c_loss))
                    loss = v_loss
                    
                    self.SToV_model.backward(SToV_loss_arr)
                    self.SToC_model.backward(SToC_loss_arr)
                    
                # VCToV with trained SToV & SToC
                elif model_type == 4:
                    SToV_outputs = self.SToV_model(seis_s).detach()
                    SToC_outputs = self.SToC_model(seis_s).detach()
                    
                    outputs = torch.cat((SToV_outputs, SToC_outputs), dim=1)
                    VCToV_outputs, VCToV_loss_arr = self.VCToV_model.process(outputs, v_s)
                    
                    # cc metrics
                    v_loss = self.metrics_s(SToV_outputs, v_s)
                    c_loss = self.metrics_s(SToC_outputs, c_s)
                    v2_loss = self.metrics_s(VCToV_outputs, v_s)
                    # print('Loss v_loss: {:.12f}, c_loss: {:.12f}, v2_loss: {:.12f}'\
                    #     .format(v_loss, c_loss, v2_loss))
                    loss = v2_loss
                    
                    self.VCToV_model.backward(VCToV_loss_arr)
                                        
                # joint
                elif model_type == 5:
                    SToV_outputs, SToV_loss_arr = self.SToV_model.process(seis_s, v_s)
                    SToC_outputs, SToC_loss_arr = self.SToC_model.process(seis_s, c_s)
                    
                    outputs = torch.cat((SToV_outputs, SToC_outputs), dim=1)
                    VCToV_outputs, VCToV_loss_arr = self.VCToV_model.process(outputs.detach(), v_s)
                    
                    # cc metrics
                    v_loss = self.metrics_s(SToV_outputs, v_s)
                    c_loss = self.metrics_s(SToC_outputs, c_s)
                    v2_loss = self.metrics_s(VCToV_outputs, v_s)
                    # print('Loss v_loss: {:.12f}, c_loss: {:.12f}, v2_loss: {:.12f}'\
                    #     .format(v_loss, c_loss, v2_loss))
                    loss = v2_loss
                    
                    self.VCToV_model.backward(VCToV_loss_arr) # stage 2 backward first!!!
                    self.SToV_model.backward(SToV_loss_arr)
                    self.SToC_model.backward(SToC_loss_arr)
                    
                epoch_loss += loss
                
            self.lr_scheduler_step()
                
            elapsed_time = time.time() - cur_time
            print('Epoch {} elapsed_time: {} = {}m {}s'.format(epoch, \
                elapsed_time, elapsed_time // 60, elapsed_time % 60))
            print('Epoch {} loss: {:.12f}'.format(epoch, epoch_loss / num_batches))
            training_time += elapsed_time
            
            epoch_loss_recorder[epoch] = epoch_loss / num_batches
            
            if (epoch + 1) % save_every_epochs == 0:
                self.save(epoch)
            
        print('training_time: {} = {}m {}s'.format(training_time, \
            training_time // 60, training_time % 60))
        
        np.save(os.path.join(self.result_path, 'loss_t{}_epochs{}_{:.0f}.npy'.format(model_type, max_epochs, time.time())), epoch_loss_recorder)
        
        # self.save()
        
    # calculate values of metrics
    def eval(self):
        print('\n################')
        print('#  Eval Phase  #')
        print('################\n')
        
        self.MODE = 2 # test mode
        
        # Get data
        seis_dataset, v_dataset, c_dataset = batch_read(config, self.MODE == 1)
        
        loader = get_DataLoader(seis_dataset, v_dataset, c_dataset, config.BATCH_SIZE, shuffle=False)
        
        # Load models
        self.load()
        
        num_batches = len(loader)
        
        # Activate the eval mode!
        self.SToV_model.eval()
        self.SToC_model.eval()
        self.VCToV_model.eval()
        
        model_type = self.config.MODEL_TYPE
        device = self.device
        
        sum_mse_loss, sum_mae_loss, sum_lpips_loss, sum_uqi_loss = 0.0, 0.0, 0.0, 0.0
        cur_time = time.time()
        with torch.no_grad():
            for items in loader:
                seis_s, v_s, c_s = items
                
                seis_s = seis_s.to(device)
                v_s = v_s.to(device)
                c_s = c_s.to(device)
                
                # SToV
                if model_type == 1:
                    SToV_outputs = self.SToV_model(seis_s)
                    
                    # cc metrics
                    v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss = self.metrics_f(SToV_outputs, v_s)
                    # print('v_mse_loss: {:.12f}, v_mae_loss: {:.12f}, v_lpips_loss: {:.12f}, v_uqi_loss: {:.12f}'\
                    #     .format(v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss))
                    mse_loss, mae_loss, lpips_loss, uqi_loss = v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss
                    
                # SToC
                elif model_type == 2:
                    SToC_outputs = self.SToC_model(seis_s)
                    
                    # cc metrics
                    c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss = self.metrics_f(SToC_outputs, c_s)
                    # print('c_mse_loss: {:.12f}, c_mae_loss: {:.12f}, c_lpips_loss: {:.12f}, c_uqi_loss: {:.12f}'\
                    #     .format(c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss))
                    mse_loss, mae_loss, lpips_loss, uqi_loss = c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss
                
                # SToV & SToC
                elif model_type == 3:
                    SToV_outputs = self.SToV_model(seis_s)
                    SToC_outputs = self.SToC_model(seis_s)
                    
                    # cc metrics
                    v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss = self.metrics_f(SToV_outputs, v_s)
                    c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss = self.metrics_f(SToC_outputs, c_s)
                    # print('v_mse_loss: {:.12f}, v_mae_loss: {:.12f}, v_lpips_loss: {:.12f}, v_uqi_loss: {:.12f}'\
                    #     .format(v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss))
                    # print('c_mse_loss: {:.12f}, c_mae_loss: {:.12f}, c_lpips_loss: {:.12f}, c_uqi_loss: {:.12f}'\
                    #     .format(c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss))
                    mse_loss, mae_loss, lpips_loss, uqi_loss = v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss
                    
                # VCToV with trained SToV & SToC | joint
                elif model_type == 4:
                    SToV_outputs = self.SToV_model(seis_s)
                    SToC_outputs = self.SToC_model(seis_s)
                    
                    outputs = torch.cat((SToV_outputs, SToC_outputs), dim=1) # cat will sum both in_channels
                    VCToV_outputs = self.VCToV_model(outputs)
                    
                    # cc metrics
                    v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss = self.metrics_f(SToV_outputs, v_s)
                    c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss = self.metrics_f(SToC_outputs, c_s)
                    v2_mse_loss, v2_mae_loss, v2_lpips_loss, v2_uqi_loss = self.metrics_f(VCToV_outputs, v_s)
                    # print('v_mse_loss: {:.12f}, v_mae_loss: {:.12f}, v_lpips_loss: {:.12f}, v_uqi_loss: {:.12f}'\
                    #     .format(v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss))
                    # print('c_mse_loss: {:.12f}, c_mae_loss: {:.12f}, c_lpips_loss: {:.12f}, c_uqi_loss: {:.12f}'\
                    #     .format(c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss))
                    # print('v2_mse_loss: {:.12f}, v2_mae_loss: {:.12f}, v2_lpips_loss: {:.12f}, v2_uqi_loss: {:.12f}'\
                    #     .format(v2_mse_loss, v2_mae_loss, v2_lpips_loss, v2_uqi_loss))
                    mse_loss, mae_loss, lpips_loss, uqi_loss = v2_mse_loss, v2_mae_loss, v2_lpips_loss, v2_uqi_loss
                    
                sum_mse_loss += mse_loss
                sum_mae_loss += mae_loss
                sum_lpips_loss += lpips_loss
                sum_uqi_loss += uqi_loss
                
        evaluating_time = time.time() - cur_time
        print('evaluating_time: {} = {}m {}s'.format(evaluating_time, evaluating_time // 60, evaluating_time % 60))
        print('ave_mse_loss: {:.12f}, ave_mae_loss: {:.12f}, ave_lpips_loss: {:.12f}, ave_uqi_loss: {:.12f}'.\
            format(sum_mse_loss / num_batches, sum_mae_loss / num_batches, sum_lpips_loss / num_batches, sum_uqi_loss / num_batches))
        
    # output the predicted data
    def test(self, sample_size=1, testID=1):
        print('\n################')
        print('#  Test Phase  #')
        print('################\n')
        
        self.MODE = 2 # test mode
        
        # Get data
        seis_dataset, v_dataset, c_dataset = batch_read(config, self.MODE == 1)
        
        # loader = get_DataLoader(seis_dataset, v_dataset, c_dataset, batch_size=1) # shuffle=True: random sample
        loader = get_DataLoader(seis_dataset, v_dataset, c_dataset, batch_size=1, shuffle=False) # sequential sample
        
        # Load models
        self.load()
        
        num_batches = len(loader)
        sample_size = min(sample_size, num_batches)
        
        # Activate the eval mode!
        self.SToV_model.eval()
        self.SToC_model.eval()
        self.VCToV_model.eval()
        
        model_type = config.MODEL_TYPE
        device = self.device
        
        iterator = iter(loader)
        
        testID = max(testID, 1) # ensure start from 1
        for _ in range(testID - 1):
            next(iterator)
            
        filen = os.path.join(self.result_path, 't{}_id{}.npz'.format(model_type, testID))
        
        sum_mse_loss, sum_mae_loss, sum_lpips_loss, sum_uqi_loss = 0.0, 0.0, 0.0, 0.0
        cur_time = time.time()
        with torch.no_grad():
            for _ in range(sample_size):
                seis_s, v_s, c_s = next(iterator)
                
                seis_s = seis_s.to(device)
                v_s = v_s.to(device)
                c_s = c_s.to(device)
                
                # SToV
                if model_type == 1:
                    SToV_outputs = self.SToV_model(seis_s)
                    
                    # cc metrics
                    v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss = self.metrics_f(SToV_outputs, v_s)
                    print('v_mse_loss: {:.12f}, v_mae_loss: {:.12f}, v_lpips_loss: {:.12f}, v_uqi_loss: {:.12f}'\
                        .format(v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss))
                    mse_loss, mae_loss, lpips_loss, uqi_loss = v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss
                    
                    v_s = v_s.cpu().detach().numpy()[0,0]
                    pred_v = SToV_outputs.cpu().detach().numpy()[0,0]
                    
                    # # v_show(pred_v)
                    # v_show(v_s, \
                    #     pred_v)
                    
                    # v_show(v_s)
                    # v_show(pred_v)
                    
                    np.savez(filen, v_s=v_s, pred_v=pred_v)
                    
                # SToC
                elif model_type == 2:
                    SToC_outputs = self.SToC_model(seis_s)
                    
                    # cc metrics
                    c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss = self.metrics_f(SToC_outputs, c_s)
                    print('c_mse_loss: {:.12f}, c_mae_loss: {:.12f}, c_lpips_loss: {:.12f}, c_uqi_loss: {:.12f}'\
                        .format(c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss))
                    mse_loss, mae_loss, lpips_loss, uqi_loss = c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss
                    
                    c_s = c_s.cpu().detach().numpy()[0,0]
                    pred_c = SToC_outputs.cpu().detach().numpy()[0,0]
                    
                    # # v_show(pred_c)
                    # v_show(c_s, \
                    #     pred_c)
                    
                    # v_show(c_s)
                    # v_show(pred_c)
                    
                    np.savez(filen, c_s=c_s, pred_c=pred_c)
                
                # SToV & SToC
                elif model_type == 3:
                    SToV_outputs = self.SToV_model(seis_s)
                    SToC_outputs = self.SToC_model(seis_s)
                    
                    # cc metrics
                    v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss = self.metrics_f(SToV_outputs, v_s)
                    c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss = self.metrics_f(SToC_outputs, c_s)
                    print('v_mse_loss: {:.12f}, v_mae_loss: {:.12f}, v_lpips_loss: {:.12f}, v_uqi_loss: {:.12f}'\
                        .format(v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss))
                    print('c_mse_loss: {:.12f}, c_mae_loss: {:.12f}, c_lpips_loss: {:.12f}, c_uqi_loss: {:.12f}'\
                        .format(c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss))
                    mse_loss, mae_loss, lpips_loss, uqi_loss = v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss
                    
                    v_s = v_s.cpu().detach().numpy()[0,0]
                    c_s = c_s.cpu().detach().numpy()[0,0]
                    pred_v = SToV_outputs.cpu().detach().numpy()[0,0]
                    pred_c = SToC_outputs.cpu().detach().numpy()[0,0]
                    
                    # # v_show(pred_v, pred_c)
                    # v_show(v_s, c_s, \
                    #     pred_v, pred_c)
                    
                    np.savez(filen, v_s=v_s, c_s=c_s, pred_v=pred_v, pred_c=pred_c)
                    
                # VCToV with trained SToV & SToC | joint
                else:
                    SToV_outputs = self.SToV_model(seis_s)
                    SToC_outputs = self.SToC_model(seis_s)
                    
                    outputs = torch.cat((SToV_outputs, SToC_outputs), dim=1) # cat will sum both in_channels
                    VCToV_outputs = self.VCToV_model(outputs)
                    
                    # cc metrics
                    v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss = self.metrics_f(SToV_outputs, v_s)
                    c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss = self.metrics_f(SToC_outputs, c_s)
                    v2_mse_loss, v2_mae_loss, v2_lpips_loss, v2_uqi_loss = self.metrics_f(VCToV_outputs, v_s)
                    print('v_mse_loss: {:.12f}, v_mae_loss: {:.12f}, v_lpips_loss: {:.12f}, v_uqi_loss: {:.12f}'\
                        .format(v_mse_loss, v_mae_loss, v_lpips_loss, v_uqi_loss))
                    print('c_mse_loss: {:.12f}, c_mae_loss: {:.12f}, c_lpips_loss: {:.12f}, c_uqi_loss: {:.12f}'\
                        .format(c_mse_loss, c_mae_loss, c_lpips_loss, c_uqi_loss))
                    print('v2_mse_loss: {:.12f}, v2_mae_loss: {:.12f}, v2_lpips_loss: {:.12f}, v2_uqi_loss: {:.12f}'\
                        .format(v2_mse_loss, v2_mae_loss, v2_lpips_loss, v2_uqi_loss))
                    mse_loss, mae_loss, lpips_loss, uqi_loss = v2_mse_loss, v2_mae_loss, v2_lpips_loss, v2_uqi_loss
                    
                    v_s = v_s.cpu().detach().numpy()[0,0]
                    c_s = c_s.cpu().detach().numpy()[0,0]
                    pred_v = SToV_outputs.cpu().detach().numpy()[0,0]
                    pred_c = SToC_outputs.cpu().detach().numpy()[0,0]
                    pred_v_last = VCToV_outputs.cpu().detach().numpy()[0,0]
                    
                    # # v_show(pred_v, pred_c, pred_v_last)
                    # v_show(v_s, c_s, v_s, \
                    #     pred_v, pred_c, pred_v_last)
                    
                    # v_show(v_s)
                    # v_show(pred_v)
                    # v_show(pred_v_last)
                    
                    np.savez(filen, v_s=v_s, c_s=c_s, v_s_repeated=v_s, \
                        pred_v=pred_v, pred_c=pred_c, pred_v_last=pred_v_last)
                    
                sum_mse_loss += mse_loss
                sum_mae_loss += mae_loss
                sum_lpips_loss += lpips_loss
                sum_uqi_loss += uqi_loss
                                                        
        evaluating_time = time.time() - cur_time
        print('evaluating_time: {} = {}m {}s'.format(evaluating_time, evaluating_time // 60, evaluating_time % 60))
        print('ave_mse_loss: {:.12f}, ave_mae_loss: {:.12f}, ave_lpips_loss: {:.12f}, ave_uqi_loss: {:.12f}'.\
            format(sum_mse_loss / sample_size, sum_mae_loss / sample_size, sum_lpips_loss / sample_size, sum_uqi_loss / sample_size))
        
if __name__ == '__main__':
    # load config
    config_path = 'config.yml'
    config = Config(config_path)
    
    # config.print()
    # print(config.BATCH_SIZE)
    
    # initialize random seed
    torch.manual_seed(config.SEED) # Set random seed for CPU operations
    torch.cuda.manual_seed_all(config.SEED) # Set random seed for GPU operations
    # np.random.seed(config.SEED)
    # random.seed(config.SEED)
    
    model = VBF(config)
    
    model.train()
    
    model.eval()
    
    # model.test() # default: testID=1
    model.test(testID=2)
    model.test(testID=8)