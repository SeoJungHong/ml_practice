import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class TitanicDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        df = pd.read_csv('./data/TITANIC/train.csv', header=0, index_col=0)
        df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

        # Fill missing data: Age and Fare with the mean, Embarked with most frequent value
        df[['Age']] = df[['Age']].fillna(value=df[['Age']].mean())
        df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())
        df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())

        # Convert categorical  features into numeric
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

        # Convert Embarked to one-hot
        enbarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = df.drop('Embarked', axis=1)
        df = df.join(enbarked_one_hot)

        self.len = df.shape[0]
        self.x_data = df.drop(['Survived'], axis=1).values.astype(float)
        self.y_data = df['Survived'].values

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(9, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 2)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.activation(self.l1(x))
        out2 = self.activation(self.l2(out1))
        y_pred = self.activation(self.l3(out2))
        return y_pred


dataset = TitanicDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

model = Model()
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

model.train()
for epoch in range(10):
    print("##### Training epoch ", epoch, " started")
    for batch_index, (inputs, labels) in enumerate(train_loader):
        inputs, labels = Variable(inputs).float(), Variable(labels)
        # inputs, labels = Variable(torch.FloatTensor(inputs)), Variable(torch.LongTensor(labels))
        # Run your training process
        print(batch_index, "inputs : ", inputs.data, "labels : ", labels.data)

        # Forward pass: Compute predicted y by passing x to the model
        outputs = model(inputs)
        # Compute and print loss
        loss = criterion(outputs, labels)
        print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Gradients
    print("##### Training epoch ", epoch, " ended")
