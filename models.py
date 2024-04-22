import os
import torch
import torch.nn as nn
from nets import EMD, Discriminator, NestedUNet
from comparison_nets import InversionNet, FCNVMB, UNet
import torch.optim as optim
from loss import AdversarialLoss, CrossEntropyBalanced
from CosineAnnealingLR import WarmupCosineLR

'''
    GenWrapper (namely ModelWrapper) refers to a model that encapsulates
    the architecture and parameters of a neural network. It can also be used
    as the generator of a GAN.
'''
class GenWrapper(nn.Module):
    def __init__(self, config, name, stage):
        super().__init__()
        
        self.config = config
        self.name = name
        self.stage = stage
                
        self.fullname = name + '_s' + str(stage)
        model_para_path = config.MODEL_PARA_PATH
        if not os.path.exists(model_para_path):
            os.makedirs(model_para_path)
        self.gen_weights_path = os.path.join(model_para_path, self.fullname + '_gen.pth')
        
        # get model
        out_dsp_dim = config.OUT_DSP_DIM
        in_dsp_dim = config.IN_DSP_DIM
        if stage == 2:
            assert name == 'EMD_VCToV', \
                'Error: Only EMD_VCToV (generator) model can be in stage 2.'
            generator = EMD(in_channels=2,
                            out_dsp_dim=out_dsp_dim,
                            stage=2,
                            residual_blocks=6,
                            use_sn_ED=False,
                            use_sn_M=True,
                            init_weights=False)
        else: # stage = 1v or 1c
            in_channels = config.IN_CHANNELS
            
            if name == 'InversionNet':
                generator = InversionNet(in_channels)
            elif name == 'FCNVMB':
                generator = FCNVMB(n_classes=1,
                            in_channels=in_channels,
                            is_deconv=True,
                            is_batchnorm=True,
                            out_dsp_dim=out_dsp_dim)
            elif name == 'UNet_SToV' or name == 'UNet_SToC':
                generator = UNet(n_classes=1,
                            in_channels=in_channels,
                            is_deconv=True,
                            is_batchnorm=True,
                            out_dsp_dim=out_dsp_dim)
            elif name == 'NestedUNet_SToV' or name == 'NestedUNet_SToC':
                generator = NestedUNet(in_channels=in_channels,
                                       in_dsp_dim=in_dsp_dim,
                                       out_dsp_dim=out_dsp_dim)
            elif name == 'EMD_SToV':
                generator = EMD(in_channels=in_channels,
                                out_dsp_dim=out_dsp_dim,
                                stage=1,
                                residual_blocks=9,
                                use_sn_ED=True,
                                use_sn_M=True,
                                init_weights=False)
                                # init_weights=True)
            elif name == 'EMD_SToC':
                generator = EMD(in_channels=in_channels,
                                out_dsp_dim=out_dsp_dim,
                                stage=1,
                                residual_blocks=6,
                                use_sn_ED=False,
                                use_sn_M=True,
                                init_weights=False)
            
        # initialize the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # move the model to the device
        generator = generator.to(self.device)
        
        # use multiple GPUs
        if torch.cuda.device_count() > 1:
            print('Using', torch.cuda.device_count(), 'GPUs.')
            generator = nn.DataParallel(generator)
            
        self.generator = generator
        
        # loss
        if name[-3:] == 'ToC':
            self.base_loss = CrossEntropyBalanced()
        else:
            self.base_loss = nn.MSELoss()
        
        warmup = 10
        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR)
        )
        self.gen_lr_scheduler = WarmupCosineLR(optimizer=self.gen_optimizer,
                                               milestones=[warmup, config.MAX_EPOCHS],
                                               warmup_iters=warmup,
                                               min_ratio=1e-7)
    
    def process(self, inputs, labels):
        # zero optimizer
        self.gen_optimizer.zero_grad()
            
        # process outputs
        outputs = self(inputs)
        gen_loss = 0
        
        # generator base loss
        gen_base_loss = self.base_loss(outputs, labels) * self.config.BASE_LOSS_WEIGHT
        gen_loss += gen_base_loss
                    
        return outputs, [gen_loss]

    def forward(self, inputs):
        outputs = self.generator(inputs)
        return outputs

    def backward(self, loss_arr):
        gen_loss = loss_arr[0]
        gen_loss.backward()
        self.gen_optimizer.step()
            
    def load(self, epoch):
        path = '{}_epoch{}.pth'.format(self.gen_weights_path[:-4], epoch)
        if os.path.exists(path):
            print('Loading {} (generator) model at epoch {}...'.format(self.fullname, epoch))
            
            if torch.cuda.is_available():
                data = torch.load(path)
            else:
                # Load onto CPU
                data = torch.load(path, map_location=lambda storage, loc: storage)

            # self.generator.load_state_dict(data) # save & load on the same kind of device
            if isinstance(self.generator, nn.DataParallel):
                self.generator.module.load_state_dict(data)
            else:
                self.generator.load_state_dict(data)
        else:
            print('Error: Cannot find {} (generator) model!\nExiting the program...\n'.format(self.fullname))
            exit(1)
        
    def save(self, epoch):
        print('\nSaving {} (generator) model at epoch {}...'.format(self.fullname, epoch))
        # torch.save(self.generator.state_dict(), '{}_epoch{}.pth'.format(self.gen_weights_path[:-4], epoch)) # save & load on the same kind of device
        if isinstance(self.generator, nn.DataParallel):
            torch.save(self.generator.module.state_dict(), '{}_epoch{}.pth'.format(self.gen_weights_path[:-4], epoch)) # remove 'module.'
        else:
            torch.save(self.generator.state_dict(), '{}_epoch{}.pth'.format(self.gen_weights_path[:-4], epoch))
        
class GANWrapper(GenWrapper):
    def __init__(self, config, name, stage):
        super().__init__(config, name, stage)
        
        # update some members
        self.fullname = name + '_s' + str(stage) + '_g' # 'g' = 'GAN'
        self.gen_weights_path = os.path.join(config.MODEL_PARA_PATH, self.fullname + '_gen.pth')
        
        # discriminator PART
        discriminator = Discriminator(in_channels=1,
                                      use_sigmoid=config.GAN_LOSS != 'hinge',
                                      init_weights=False)
        
        discriminator = discriminator.to(self.device)
        
        if torch.cuda.device_count() > 1:
            discriminator = nn.DataParallel(discriminator)
            
        self.discriminator = discriminator
        
        # loss
        # GAN loss
        self.adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        
        # self.fm_loss = nn.L1Loss()
        self.fm_loss = nn.MSELoss()
        
        warmup = 10
        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR)
        )
        self.dis_lr_scheduler = WarmupCosineLR(optimizer=self.dis_optimizer,
                                               milestones=[warmup, config.MAX_EPOCHS],
                                               warmup_iters=warmup,
                                               min_ratio=1e-7)
    
    def process(self, inputs, labels):
        self.dis_optimizer.zero_grad()
        
        outputs, loss_arr = super().process(inputs, labels)
        gen_loss = loss_arr[0]
        
        dis_loss = 0
        
        # discriminator loss
        dis_input_real = labels
        dis_input_fake = outputs.detach()
        dis_real, dis_real_feat = self.discriminator(dis_input_real)
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        
        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss
        
        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.fm_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss
        
        return outputs, [gen_loss, dis_loss]

    def forward(self, inputs):
        outputs = self.generator(inputs)
        return outputs

    def backward(self, loss_arr):
        gen_loss, dis_loss = loss_arr
        
        dis_loss.backward() # discriminator first!!!
        gen_loss.backward()
        
        self.dis_optimizer.step()
        self.gen_optimizer.step()