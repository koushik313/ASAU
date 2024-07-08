#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class asau(nn.Module):
    def __init__(self):
        super(asau,self).__init__()

        #self.w0 = nn.Parameter(torch.tensor(0.01))
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        

    def forward(self,x):
        self.w0 = 0.01
        return self.w0 * x + ((1.0-self.w0) * x * torch.tanh(self.w2 * F.softplus((1.0-self.w0) * self.w1 * x)))
    

