import torch
import time
import pandas as pd

class MLP(torch.nn.Module):
    def __init__(self, n_epochs=0, train_loader=None, test_loader=None, checkpoint_path=None):
        super(MLP,self).__init__()
        self.checkpoint_path = checkpoint_path
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lossMin = 1
        self.begin_time = 0

        self.fc1 = torch.nn.Linear(10*14*2,256)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256,128)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(128,10)
        
    def forward(self,x):
        x = x.view(-1,10*14*2)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def train(self):
        epoch_acc = []
        lossfunc = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        self.begin_time = time.time()
        for epoch in range(self.n_epochs):
            train_loss = 0.0
            for data,target in self.train_loader:
                optimizer.zero_grad()   # 清空上一步的残余更新参数值
                output = self.forward(data)    # 得到预测值
                loss = lossfunc(output,target)  # 计算两者的误差
                loss.backward()         # 误差反向传播, 计算参数更新值
                optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
                train_loss += loss.item()*data.size(0)
            train_loss = train_loss / len(self.train_loader.dataset)
            self.lossMin = min(self.lossMin, train_loss)
            print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
            epoch_acc.append(self.test())
            self.save_checkpoint(epochID=epoch, optimizer=optimizer, lossfunc=lossfunc)
        name=['Accuracy']
        test=pd.DataFrame(columns=name,data=epoch_acc)#数据有三列，列名分别为one,two,three
        test.to_csv('./acc.csv',encoding='gbk')
            

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                outputs = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
        return 100.0 * correct / total

    def save_checkpoint(self, epochID, optimizer, lossfunc):
        torch.save({'epoch': epochID + 1, 'state_dict': self.state_dict(), 'best_loss': self.lossMin},
                           self.checkpoint_path + '/m-' + str(self.begin_time) + '-epoch'+str(epochID + 1)+'-' + str("%.4f" % self.lossMin) + '.pth')

    def load_checkpoint(self, subpath):
        model_CKPT = torch.load(self.checkpoint_path+"/"+subpath)
        self.load_state_dict(model_CKPT['state_dict'])
        print('loaded checkpoint')