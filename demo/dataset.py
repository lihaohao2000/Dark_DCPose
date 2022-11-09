import torch
 
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform = None):
        self.data = data
        self.target = target
        self.transform = transform
 
    def __getitem__(self, index):
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, self.target[index]
 
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.data)
