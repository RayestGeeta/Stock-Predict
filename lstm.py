#!/usr/bin/env python
# coding: utf-8

# In[21]:


# 导入包
import pandas as pd
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset, TensorDataset

# 读取数据
data = pd.read_csv('suning.csv', encoding = 'gbk')
# 删除空值的行
data = data.drop(data[data['涨跌额'].str.contains('None')].index)


# In[48]:


# 把数据设置成统一的float数据类型
data["涨跌额"] = data['涨跌额'].astype("float")
data["最高价"] = data['最高价'].astype("float")
data["最高价"] = data['最高价'].astype("float")
data["开盘价"] = data['开盘价'].astype("float")
data["涨跌幅"] = data['涨跌幅'].astype("float")
data["换手率"] = data['换手率'].astype("float")


# 导入数据和预测值
datas = data[data.columns[4:7]].values
labels = data[data.columns[3]].values




# In[74]:


dataset = TensorDataset(torch.tensor(np.array(datas.reshape(-1, 3))[3000:]), torch.tensor(np.array(labels))[3000:])
dataloader = DataLoader(dataset, batch_size=len(dataset),shuffle=True, drop_last=False)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
    
    
        self.lstm = nn.LSTM(1, 2,2)
        self.linear = nn.Linear(3, 10)
        self.linear1 = nn.Linear(10,8)
        self.linear3 = nn.Linear(8,2)
        self.linear4 = nn.Linear(2,1)

    
    def forward(self,x):
        x1 = x.clone()
        x1,_ = self.lstm(x1.reshape(-1 ,3,1))
        a,b,c = x1.shape
        out = self.linear4(x1.view(-1,c))
        out1 = out.view(a,b,-1)
        out = self.linear(x)
        out = self.linear1(out)
        out = self.linear3(out)
        out = self.linear4(out)

        
        
        return out


rnn = LSTM()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)   # optimize all cnn parameters
loss_func = nn.MSELoss()

#h_state = None      # for initial hidden state


for epoch in range(20):
    for step,(b_x, b_y) in enumerate(dataloader):


        prediction = rnn(b_x.float())   # rnn output
        # !! next step is important !!
        #h_state = h_state.data        # repack the hidden state, break the connection from last iteration

        #print(prediction, b_y.float())
        loss = loss_func(prediction, b_y.float())         # calculate loss
        #print(loss.data.numpy())
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients

plt.plot(rnn(b_x.float()).view(-1).data.numpy()[:50])
plt.plot(b_y[:50].data.numpy())


# In[ ]:




