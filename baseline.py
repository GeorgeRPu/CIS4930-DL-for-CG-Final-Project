import matplotlib.pyplot as plt
import models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST('.data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1)
net = models.CNN(1, 28, 28, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

testset = torchvision.datasets.MNIST('.data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4)

accuracies = []
for epoch in range(20):
    print('Epoch', epoch)
    for images, labels in tqdm.tqdm(trainloader):
        net.zero_grad()
        out = net(images.to(device))
        loss = criterion(out, labels.to(device))
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    print(f'Test accuracy {accuracy:.2f} %')

plt.figure(1)
plt.plot(accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Testing...')
plt.savefig('baseline.png')
