'''
Auther: Jialun Cai
2022/09/06
'''
from locale import normalize
import os
from pickletools import optimize
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from resnet import ResNet

model_name = input("Please input the name of the model:   ")
gpu = input("Please input the gpu id:   ")

os.environ['CUDA_VISIBLE_DEVICES'] = gpu

root_train = './dataset/train'
root_test = './dataset/val'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
lr = 0.0001
step_size = 5
epoch = 50
min_acc = 0


#归一化到[0,1]，不一定必要
normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])

train_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomVerticalFlip(),
                                    #   transforms.ColorJitter(brightness=0.5), #亮度
                                    #   transforms.ColorJitter(hue=0.5),  #对比度
                                    #   transforms.ColorJitter(contrast=0.5) #色调
                                      transforms.ToTensor(),
                                      normalize
                                      ])

test_transform = transforms.Compose([transforms.Resize((224,224)),
                                    #   transforms.RandomVerticalFlip(),
                                    #   transforms.ColorJitter(brightness=0.5), #亮度
                                    #   transforms.ColorJitter(hue=0.5),  #对比度
                                    #   transforms.ColorJitter(contrast=0.5) #色调
                                      transforms.ToTensor(),
                                      normalize
                                      ])

train_dataset = ImageFolder(root_train, transform=train_transform)
test_dataset = ImageFolder(root_test, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)



model = ResNet().to(device)

loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=True)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma=0.5)



def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0 ,0
    for batch, (x,y) in enumerate(tqdm(dataloader, 0)):
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output,y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y==pred)/output.shape[0]

        #反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n += 1
    train_loss = loss / n
    train_acc = current / n
    print('train_loss:     ' + str(train_loss))
    print('train_acc:     ' + str(train_acc))
    return  train_loss,train_acc

def val(dataloader, model, loss_fn):
    # 将模型转化为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(tqdm(dataloader, 0)):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    print('val_loss' + str(val_loss))
    print('val_acc' + str(val_acc))
    return val_loss, val_acc



train_loss = []
train_acc = []
test_loss = []
test_ = []

for t in range(epoch):
    # lr_scheduler.step()
    print(f'-----------------epoch{t+1}----------------------')
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = val(test_dataloader, model, loss_fn)
    lr_scheduler.step()

    if test_acc > min_acc:
            folder = "save_model/" + model_name
            if not os.path.exists(folder):
                os.mkdir(folder)
            min_acc = test_acc
            print(f"save best model, 第{t+1}轮")
            torch.save(model.state_dict(), folder + "/best_model.pth")
    print('Done！')

        