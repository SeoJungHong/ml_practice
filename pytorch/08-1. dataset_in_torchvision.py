import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

dataset = torchvision.datasets.CIFAR10(root='./data/CIFAR10',
                                       train=True,
                                       download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.5, 0.5, 0.5),
                                                                                          (0.5, 0.5, 0.5))]))
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Run your training process
        print(epoch, i, "inputs", inputs.data, "labels", labels.data)
