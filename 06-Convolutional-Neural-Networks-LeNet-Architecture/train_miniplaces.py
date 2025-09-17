# python imports
import os
import argparse
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms

from dataloader import MiniPlaces
from student_code import LeNet, train_model, test_model


def save_checkpoint(state, is_best,
                    file_folder="./outputs/",
                    filename='checkpoint.pth.tar'):
    """save checkpoint"""
    if not os.path.exists(file_folder):
        os.makedirs(os.path.expanduser(file_folder), exist_ok=True)
    torch.save(state, os.path.join(file_folder, filename))
    if is_best:
        # skip the optimization state
        state.pop('optimizer', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


# main function for training and testing
def main(args):
    # set up random seed
    torch.manual_seed(0)

    ###################################
    # setup model, loss and optimizer #
    ###################################
    model = LeNet()

    training_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optim.Adam(model.parameters(), lr=args.lr)

    # set up transforms to transform the PIL Image to tensors
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    ################################
    # setup dataset and dataloader #
    ################################
    data_folder = './data'
    if not os.path.exists(data_folder):
        os.makedirs(os.path.expanduser(data_folder), exist_ok=True)

    train_set = MiniPlaces(
        root=data_folder, split="train", download=True, transform=train_transform)
    test_set = MiniPlaces(
        root=data_folder, split="val", download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)
    
    ################################
    # start the training           #
    ################################
    # resume from a previous checkpoint
    best_acc = 0.0
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{:s}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            # load model weight
            model.load_state_dict(checkpoint['state_dict'])
            # load optimizer states
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}, acc {:0.2f})".format(
                args.resume, checkpoint['epoch'], 100*best_acc))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # training of the model
    print("Training the model ...\n")
    for epoch in range(start_epoch, args.epochs):
        # train model for 1 epoch
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        # evaluate the model on test_set after this epoch
        acc = test_model(model, test_loader, epoch)
        # save the current checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc' : max(best_acc, acc),
            'optimizer' : optimizer.state_dict(),
            }, (acc > best_acc))
        best_acc = max(best_acc, acc)
    print("Finished Training")

    # store accuracy results
    results = []

    # default configuration
    print("\nTraining with default configuration:")
    train_main(args)
    best_model_path = os.path.join(args.output_folder, 'model_best.pth.tar')
    results.append(test_with_eval_script(best_model_path))

    # configurations
    configurations = [
        {'batch_size': 8},
        {'batch_size': 16},
        {'lr': 0.05},
        {'lr': 0.01},
        {'epochs': 20},
        {'epochs': 5},
    ]

    for config in configurations:
        # update the arguments 
        for key, value in config.items():
            setattr(args, key, value)

        # run training
        print(f"\nTraining with config: {config}")
        train_main(args)

        # test eval model
        best_model_path = os.path.join(args.output_folder, 'model_best.pth.tar')
        accuracy = test_with_eval_script(best_model_path)
        results.append(accuracy)

        # save results to file
        with open('results.txt', 'w') as file:
            for accuracy in results:
                file.write(f'{accuracy:.2f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification using Pytorch')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='number of images within a mini-batch')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--output-folder', default='./outputs/', type=str,
                        help='folder to save output files (default: ./outputs/)')
    args = parser.parse_args()
    main(args)
