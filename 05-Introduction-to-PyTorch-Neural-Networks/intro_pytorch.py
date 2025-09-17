import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.FashionMNIST(root='./data', train=training, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True if training else False)
    return dataloader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    #Connect flatten and linear layers to the model
    model = nn.Sequential(nn.Flatten(), nn.Linear(784,128), nn.ReLU(), nn.Linear(128,10))
    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        model.train()
        t_loss = 0
        correct = 0
        t_samples = 0
        for batch_i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            #Accuracy
            _, predicted = output.max(1)
            t_samples += target.size(0)
            correct += predicted.eq(target).sum().item()
            t_loss += loss.item()
        accuracy = 100.0*correct/t_samples
        #Avg loss per epoch
        avg_loss = t_loss/(batch_i + 1)
        #Output training status after every epoch
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{t_samples} ({accuracy:.2f}%) Loss: {avg_loss:.3f}')

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    t_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            t_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    t_loss /= len(test_loader.dataset)
    accuracy = 100.0*correct/len(test_loader.dataset)
    if show_loss:
        #Output loss - four decimal places
        print(f'Average loss: {t_loss:.4f}')
    #Output accuracy - two decimal places (%)
    print(f'Accuracy: {accuracy:.2f}%')
    
def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    model.eval()
    with torch.no_grad():
        output = model(test_images[index].unsqueeze(0))
        prob = F.softmax(output, dim=1)
        top_prob, top_i = torch.topk(prob, 3)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
        for i in range(3):
            label = top_i[0][i].item()
            prob = top_prob[0][i].item()
            class_name = class_names[label]
            print(f'{class_name}: {prob * 100:.2f}%')

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
