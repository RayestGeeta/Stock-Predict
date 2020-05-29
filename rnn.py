#!/usr/bin/env python
# coding: utf-8

# In[20]:


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


# In[21]:


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




# In[28]:


# 将数据导入 torch的数据集数据类型
dataset = TensorDataset(torch.tensor(np.array(datas.reshape(-1, 3))[3000:]), torch.tensor(np.array(labels))[3000:])
# 将dataset导入dataloader（来进行批训练）
dataloader = DataLoader(dataset, batch_size=len(dataset),shuffle=True, drop_last=False)



class rnn(nn.Module):
    def __init__(self):#面向对象中的继承
        super(rnn, self).__init__()
    
        # rnn 网络层
        self.rnn = nn.RNN(1, 2,2)
        # 全连接层
        self.linear = nn.Linear(3, 10)
        self.linear1 = nn.Linear(10,8)
        self.linear3 = nn.Linear(8,2)
        self.linear4 = nn.Linear(2,1)

    
    def forward(self,x):
        x1,_ = self.rnn(x.reshape(-1 ,3,1))
        a,b,c = x1.shape
        out = self.linear4(x1.view(-1,c))#因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
        out1 = out.view(a,b,-1)#因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
        # 全连接层
        out = self.linear(x)
        out = self.linear1(out)
        out = self.linear3(out)
        out = self.linear4(out)

        
        
        return out

# 构建模型
rnn = rnn()
print(rnn)

# 设定优化器和误差函数
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)   # optimize all cnn parameters
loss_func = nn.MSELoss()


# 一共运行20轮
for epoch in range(20):
    # 从dataloader读取数据
    for step,(b_x, b_y) in enumerate(dataloader):


        prediction = rnn(b_x.float())   # rnn的输出

        loss = loss_func(prediction, b_y.float())         # 计算误差
        #print(loss.data.numpy())
        optimizer.zero_grad()                   # 梯度清0
        loss.backward()                         # 误差反向传播
        optimizer.step()                        # 误差更新

# 画图
plt.plot(rnn(b_x.float()).view(-1).data.numpy()[:50])
plt.plot(b_y[:50].data.numpy())


# In[ ]:




