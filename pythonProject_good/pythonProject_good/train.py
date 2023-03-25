import torch
from torch import nn
from nets import MyAlexNet
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['simHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT_TRAIN = r'./dataset/train'
ROOT_TEST = r'./dataset/val'

normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyAlexNet().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print('train_loss==' + str(train_loss))
    print('train_acc' + str(train_acc))
    return train_loss, train_acc


def val(dataloader, model, loss_fn):
    loss, current, n = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
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
    print('val_loss=' + str(val_loss))
    print('val_acc=' + str(val_acc))
    return val_loss, val_acc


def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的loss值对比图")
    plt.show()


def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.xlabel('acc')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的acc值对比图")
    plt.show()


loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch = 5
min_acc = 0
for t in range(epoch):
    lr_scheduler.step()
    print(f"epoch{t + 1}\n----------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(val_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    if val_acc > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = val_acc
        print(f"save best model，第{t + 1}轮")
        torch.save(model.state_dict(), 'save_model/best_model.pth')

    if t == epoch - 1:
        torch.save(model.state_dict(), 'save_model/best_model.pth')

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)

print('done')
