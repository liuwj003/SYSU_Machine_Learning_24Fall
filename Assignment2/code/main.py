import torch
from argument import args
from preprocess import *
from model import *
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def train(net, epoch, train_loader, criterion, optimizer, device):
    net.train()
    training_loss = 0.0
    correct = 0
    total = 0

    # for inputs, labels in train_loader:
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # inputs: [batch_size, input_size]
        # labels: [batch_size]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # zero the grad
        outputs = net(inputs)  # forward
        loss = criterion(outputs, labels)  # calculate loss
        loss.backward()  # backward, calculate the grads
        optimizer.step()  # update model's parameters using the grads

        # now, we have outputs: [batch_size, 10], dim=1: a Tensor with length=10
        max_values, indices = torch.max(outputs, 1)
        predicts = indices

        training_loss += loss.item()  # item(): Tensor(loss_value) -> loss_value

        correct_elements = torch.eq(labels, predicts)  # get bool Tensor
        correct += torch.sum(correct_elements).item()  # use item(): Tensor(num) -> num
        total += labels.size(0)
        if batch_idx % 100 == 0:
            print(f"epoch:{epoch}, batch:{batch_idx}, loss of one batch:{loss.item()}")

    accuracy = correct / total
    return training_loss, accuracy


def evaluate(net, test_loader, criterion, device):
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            max_values, predicts = torch.max(outputs, 1)
            # correct_elements = torch.eq(predicts, labels)
            correct_elements = (predicts == labels)
            correct += torch.sum(correct_elements).item()
            total += labels.size(0)

    accuracy = correct / total
    return test_loss, accuracy


def plot(train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, model_name, optimizer_name):

    # Plot the training and testing loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_list, label='Train Loss', color='blue')
    plt.plot(test_loss_list, label='Test Loss', color='red')
    plt.title(f'{model_name}-{optimizer_name}-Loss-per-batch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{model_name}_{optimizer_name}_loss.png")
    plt.show()

    # Plot the training and testing accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracy_list, label='Train Accuracy', color='green')
    plt.plot(test_accuracy_list, label='Test Accuracy', color='orange')
    plt.title(f'{model_name}-{optimizer_name}-Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{model_name}_{optimizer_name}_accuracy.png")
    plt.show()


def main():
    arguments = args()
    model = arguments.model
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    lr = arguments.learning_rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # preparing data
    X_train, Y_train, X_test, Y_test = load_data("E:/2024-1/ML/Assignment2/data")
    X_train = standardization(X_train)
    X_test = standardization(X_test)
    X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor = reshape(model, X_train, Y_train, X_test, Y_test)

    # packaging data into TensorDataset for training
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    # DataLoader for batching and shuffling data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model == 'Softmax':
        net = SoftmaxNN(3072, 10)
    elif model == 'MLP':
        net = MLP(3072, 10)
    elif model == 'MLP2':
        net = MLP2(3072, 10)
    elif model == 'MLP3':
        net = MLP3(3072, 10)
    elif model == 'MLP4':
        net = MLP4(3072, 10)
    elif model == 'CNN':
        net = CNN(10)
    elif model == 'CNN2':
        net = CNN2(10)
    elif model == 'CNN3':
        net = CNN3(10)
    elif model == 'CNN4':
        net = CNN4(10)
    elif model == 'CNN5':
        net = CNN5(10)
    else:
        raise ValueError("Invalid Model Type")

    criterion = nn.NLLLoss()

    if arguments.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif arguments.optimizer == 'Momentum':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    elif arguments.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters())  # default lr=1e-3, betas=(0.9, 0.999)
    else:
        raise ValueError("Invalid Optimizer Type")

    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for epoch in range(epochs):
        # training the model
        train_loss, train_accuracy = train(net, epoch, train_loader, criterion, optimizer, device)
        train_loss = train_loss / (X_train_tensor.size(0) / batch_size)
        print(f"epoch:{epoch}, Average train_loss per batch:{train_loss}, train_accuracy:{train_accuracy}")
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)

        # evaluating the model
        test_loss, test_accuracy = evaluate(net, test_loader, criterion, device)
        test_loss = test_loss / (X_test_tensor.size(0) / batch_size)
        print(f"epoch:{epoch}, Average test_loss per batch:{test_loss}, test_accuracy:{test_accuracy}")
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

    plot(train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, model, arguments.optimizer)


if __name__ == '__main__':
    main()