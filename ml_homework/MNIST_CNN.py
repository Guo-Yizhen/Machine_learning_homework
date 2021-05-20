import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
from tqdm import tqdm
import time

BATCH_SIZE=128
LR=0.0009
EPOCH = 5
DOWNLOAD_MNIST =False

normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

train_data = torchvision.datasets.MNIST(root='C:/Users/GYZ/Desktop/vscode/dataset/',
            train=True,
            transform=transform,   #torchvision.transforms.ToTensor()
            download=DOWNLOAD_MNIST,)
test_data = torchvision.datasets.MNIST(root='C:/Users/GYZ/Desktop/vscode/dataset/',
            train=False,
            transform=transform,   #torchvision.transforms.ToTensor()
            download=DOWNLOAD_MNIST,)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)      #batch_size为一次训练所选样本数
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        ''' 
            in_channel代表输入数据的个数，将二维像素数据作为一个数据输入. out_channel表述输出后的数据层数，超参数，可乱写  
            padding=kernal_size/2+1 对边缘进行包裹
            kernalsize 为卷积核的选取数据量
            stride 为步长
            Pool池化 kernel_size为变为几分之一的图像大小
        '''
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,padding=1,stride=1,kernel_size=3),     
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2)
                                 )
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=16,out_channels=32,padding=1,stride=1,kernel_size=3),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2)
                                 )
        self.dense=nn.Linear(32*7*7,128)
        self.predict=nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x= self.dense(x)
         #reshape
        output = self.predict(x)
        return output

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), LR) #LR为learning rate
loss_func = nn.CrossEntropyLoss()

test_y=test_data.test_labels[:2000]
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000]/255.0





for epoch in range(EPOCH):
    for step,(batch_x,batch_y) in enumerate(train_loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        output = cnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0:              #每50个step进行一次测试输出测试效果
            test_output = cnn(test_x)
            # squeeze将维度值为1的除去，例如[64, 1, 28, 28]，变为[64, 28, 28]
            pre_y = torch.max(test_output, 1)[1].data.squeeze()
            # 总预测对的数除总数就是对的概率
            accuracy = float((pre_y == test_y).sum()) / float(test_y.size(0))
            print("epoch:", epoch, "| train loss:%.4f" % loss.data, "|test accuracy：%.4f" % accuracy)

test_output = cnn(test_x)
pre_y = torch.max(test_output, 1)[1].data.squeeze()
accuracy = float((pre_y == test_y).sum()) / float(test_y.size(0))
print("test accuracy：%.4f" % accuracy)