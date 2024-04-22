import torch
import torch.nn as nn
import lpips
import numpy as np
from scipy.ndimage import uniform_filter

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            labels = labels.to('cuda')
            loss = self.criterion(outputs, labels)
            return loss
        
class Metrics_short(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.crit_mse = nn.MSELoss()
        
    def __call__(self, outputs, labels):
        mse = self.crit_mse(outputs, labels)
        
        return mse.item()
    
def _uqi_single(GT, P, ws):
    N = ws**2
    window = np.ones((ws, ws))
    
    GT_sq = GT * GT
    P_sq = P * P
    GT_P = GT * P

    GT_sum = uniform_filter(GT, ws)
    P_sum = uniform_filter(P, ws)
    GT_sq_sum = uniform_filter(GT_sq, ws)
    P_sq_sum = uniform_filter(P_sq, ws)
    GT_P_sum = uniform_filter(GT_P, ws)

    GT_P_sum_mul = GT_sum * P_sum
    GT_P_sum_sq_sum_mul = GT_sum * GT_sum + P_sum * P_sum
    numerator = 4 * (N * GT_P_sum - GT_P_sum_mul) * GT_P_sum_mul
    denominator1 = N * (GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
    denominator = denominator1 * GT_P_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0), (GT_P_sum_sq_sum_mul != 0))
    q_map[index] = 2 * GT_P_sum_mul[index] / GT_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index] / denominator[index]

    s = int(np.round(ws/2))
    return np.mean(q_map[s:-s, s:-s])

def uqi(GT, P, ws=8):
    '''
        calculate universal image quality index (uqi)
        usage:
            uqi(img1, img2)
        param ws: sliding window size
        returns: float -- uqi value
    '''
    
    return np.mean([_uqi_single(GT[index], P[index], ws) for index in np.ndindex(GT.shape[:-2])]) # GT's shape: [..., h, w]
      
class Metrics_full(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.crit_mse = nn.MSELoss()
        self.crit_mae = nn.L1Loss()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.crit_lpips = lpips.LPIPS(net='alex', version='0.1').to(device)
        
    def __call__(self, outputs, labels):
        mse = self.crit_mse(outputs, labels)
        mae = self.crit_mae(outputs, labels)
        lpips_vi = self.crit_lpips(outputs, labels).mean()
        uqi_v = uqi(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
        
        return mse.item(), mae.item(), lpips_vi.item(), uqi_v
    
class CrossEntropyBalanced(nn.Module):
    def __init__(self):
        super(CrossEntropyBalanced, self).__init__()

    def __call__(self, logits_s, labels):
        '''
            Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
            Compute edge pixels for each training sample and set as pos_weights to
            torch.nn.functional.binary_cross_entropy_with_logits
        '''
        n = logits_s.shape[0]
        
        logits_s_flat = logits_s.view(n, -1)
        labels_flat = labels.view(n, -1)
        
        y = labels_flat.float()
        
        count_neg = (1. - y).sum(dim=1)
        count_pos = y.sum(dim=1)
        
        # Equation [2]
        beta = count_neg / (count_neg + count_pos)
        
        # Equation [2] divide by 1 - beta
        pos_weight = beta / (1 - beta)
        
        '''
            Difference compared to 'SigmoidCrossEntropyBalanced'.
        '''
        weight = torch.ones_like(labels_flat)
        # weight = torch.where(labels_flat == 1, pos_weight.view(-1, 1), weight)
        weight = torch.where(labels_flat > 0.5, pos_weight.view(-1, 1), weight) # robust
        
        loss = nn.functional.binary_cross_entropy(logits_s_flat, labels_flat, weight=weight) # doesn't do sigmoid
        '''
            Difference ends.
        '''
        
        # Multiply by 1 - beta
        loss = torch.mean(loss * (1 - beta))
        
        # check if image has no edge pixels return 0 else return complete error function
        return torch.where(count_pos == 0.0, torch.tensor(0.0).to(loss.device), loss).mean()