import torch
import os
import time
import copy
import torchvision
import torchvision.datasets as datasets
import argparse
from torchvision .transforms import ToTensor
import torchvision.transforms as transforms
from torch import nn
from quant_acts import *
from fold_batch_norm import *
from quant_weis import *

def mystr2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

#定义参数
parser = argparse.ArgumentParser(description='PyTorch PWLQ code on CIFAR-10')
parser.add_argument('-wb','--wei_bits',default = 3.0,type = float,metavar = 'WB',help='weight quantization bits')
parser.add_argument('-sb','--scale_bits',default = 0.0, type=float,metavar='SB',help='scale/shift quantization bits')
parser.add_argument('-bc','--bias_corr',default=True,type=mystr2bool,help = 'Whether to use bias correction for weights quantization')
parser.add_argument('-appx','--approximate',default=True ,type=mystr2bool,help = 'Whether to use approximated optimal breakpoint')
parser.add_argument('-bkp','--break_point',default='norm',type = str, help = 'how to get optimal breakpoint:norm,laplace,search')
parser.add_argument('-wq' , '--wei_quant_scheme', default= 'pw-2', type=str, choices=['uniform','pw-2','pw-1'],help='weight quantization scheme: uniform,PWLQ')
def main():
    args = parser.parse_args()
    print(str(args))
    print()

    #模型准备
    path = r'C:\Users\86137\OneDrive\桌面\paper\AI\quantization\复现\PTQ\PWLQ-master\model\resnet50_CIFAR10_87_3.pt'
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.load_state_dict(torch.load(path))
    checkpoint = resnet50.state_dict()
    print('----- pretrained model loaded -----')

#    resnet50.to("cuda")

    #数据集准备
    train_data = datasets.CIFAR10(
        root = "data",
        train = True,
        download = True,
        transform=ToTensor()   #此处可加数据增强
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()  #可增加数据增强
    )

    train_dataloader = torch.utils.data.DataLoader(train_data ,batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True)

    critirion = nn.CrossEntropyLoss()

    #fold_batch_normalizaiton 将训练后的得到模型的batch_normalization的参数折叠进权重中
    checkpoint, weight_layers = fold_batch_norm(checkpoint)

    #quantizate weights 量化权重
    rmse = 0
    print('quantize weights ...')
#    checkpoint, rmse = quant_checkpoint(checkpoint, weight_layers, args)

    resnet50.load_state_dict(checkpoint)
    #quntization activations 量化激活层
    resnet50 = quant_model_acts(resnet50,4.0,True,4)
    validate(train_dataloader,resnet50,critirion)
# 定义训练函数
def train(dataset, dataloader, model, lr, num_epochs=20, device='gpu'):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    size = len(dataloader.dataset)

    best_model_wts = copy.deepcopy(model.state_dict())  # 记录最佳模型参数
    best_acc = 0.0

    for epoch in range(num_epochs):  # epoch信息
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # 开启训练模式
        running_loss = 0.0
        running_corrects = 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to("cuda"), y.to("cuda")

            optimizer.zero_grad()
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, y)

            # 反向传播
            loss.backward()
            optimizer.step()  # 更新权重

            # 训练损失
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds == y.data)

        #            print('Loss:{},Corrects:{},Batch:{}/64'.format(running_loss,running_corrects,len(dataloader)))
        #       scheduler.step()     #更新学习率

        epoch_loss = running_loss / len(dataloader)  # 整个epoch中的loss
        epoch_acc = running_corrects.double() / len(dataloader)  # 准确率
        print('Epoch:{} epoch_loss:{} epoch_acc:{}'.format(epoch, epoch_loss, epoch_acc))
        if (epoch_acc > best_acc):
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    print('Best val Acc:{:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)


# 定义测试函数
def test(dataloader, model):
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0.1
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cuda"), y.to("cuda")
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
#            if args.gpu is not None:
#            images = images.to("cuda")
#            target = target.to("cuda")

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

#            if args.get_stats and (i + 1) * args.batch_size >= 512:
#                break

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('total running time: %.2f min\n' % ((end_time - start_time) / 60))