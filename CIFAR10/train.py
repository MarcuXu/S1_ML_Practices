import os
import os.path as osp
# third-party packages
import pyprind
import glog as log
# pytorch related packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
import argparse
# Model Definition
from model import Net
from cifar_dataset import CIFAR10Dataset
import torch.utils.data
# Training Part
# Please fill the training part based on the given model/dataloader/optimizer/criterion


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, data in enumerate(
            pyprind.prog_bar(
                train_loader,
                title="[Epoch {}: Training]".format(epoch),
                width=40,
            )):

        def one_iteration(model, data, target, criterion):
            '''
            Please fill the training iteration with given components:
            model: our provided convolutional neural network
            data:  Images
            target: category of the images
            criterion: the loss function
        '''
            output = F.softmax(model(data), dim=1)
            loss = criterion(output, target)
            loss.backward()

        data, target = data['image'], data['label']
        optimizer.zero_grad()
        one_iteration(model, data, target, criterion)
        optimizer.step()


# Testing Part
def test(args, model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data, target = data['image'], data['label']
            output = model(data)
            pred = output.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    log.info('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, default=None, help="Model Path.")
    parser.add_argument('--model-name',
                        type=str,
                        default='char_cnn.pt',
                        help='Trained model name (defaut: char_cnn.pt).')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Fill the data directory: [train] and [test] should be at this path:
    data_dir = '/home/denglong/workspace/processed/processed'
    # data_dir = 'D:\document\google_download\homework3\homework3-students\cifar10\cifar-10-batches-py'
    # write your own dataloader to read images and targets from [data_dir]
    # Then initialize your own [train_loader], [val_loader] and [num_classes]
    # you can check torch.utils.data.DataLoader for help
    ############################
    num_classes = 10  # number of categories
    train_loader = torch.utils.data.DataLoader(
        CIFAR10Dataset('./cifar-10-batches-py', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)  # training set loader
    test_loader = torch.utils.data.DataLoader(
        CIFAR10Dataset('./cifar-10-batches-py', split='test'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)  # testing set loader
    #############################
    # get the model definition
    model = Net(num_classes=num_classes)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    lr = args.lr
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0.001,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        lr_schedule = (epoch >= 0.8 * (args.epochs))
        """
            replace False with the condition for lr_schedule
            we want to reduce the learning rate at the 80% of the training process
        """
        if lr_schedule:
            lr = lr * 0.1
            print("[Learning Rate] {}".format(lr))
            """
                schedule the learning rate used in optimizer
            """

        train(args, model, train_loader, optimizer, criterion, epoch)
        test(args, model, test_loader, epoch=epoch)

    # saving the trained model and category names
    result = {
        'state_dict': model.state_dict(),
        # 'classes': trainset.classes,
        # 'class_to_idx': trainset.class_to_idx,
    }
    torch.save(result, "char_cnn.pt")
    print("Trained Model saved to: {}".format('./char_cnn.pt'))


if __name__ == '__main__':
    main()
