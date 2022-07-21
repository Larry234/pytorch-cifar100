import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class ConcatPooling2d(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, same=False):
        super(ConcatPooling2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same
    
    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)

            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)

        else:
            padding = self.padding
            
        return padding
    
    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        B, C, H, W = x.size()[:4]
        x = x.contiguous().view(B, C, H, W, -1)
        y = x[:, :, :, :, 0]
        for i in range(x.size()[-1] - 1):
            y = torch.cat((y, x[:, :, :, :, i+1]), 1)
        return y
            
        
class KMaxPooling2d(nn.Module):
    def __init__(self, kernel_size, stride, k, padding=0, same=False):
        super(KMaxPooling2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.k = k
        self.same = False
    
    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)

            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)

        else:
            padding = self.padding
            
        return padding
            
    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        B, C, H, W = x.size()[:4]
        x = x.contiguous().view(B, C, H, W, -1)
        x, indice = torch.sort(x, dim=-1, descending=True, stable=True)
        y = x[:, :, :, :, 0]
        # select Top k values
        for i in range(self.k - 1):
            y = torch.cat((y, x[:, :, :, :, i+1]), 1)
        return y
    
