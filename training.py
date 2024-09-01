from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import numpy as np
from CIFAKEClassifier import CIFAKEClassifier
from device import fetchDevice
import dataloader 
from dataloader import CIFAKEDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os








class DatasetClassifier:
    def __init__(self, fakeFolder):
        self.fakeFolder = fakeFolder
        self.data_set = CIFAKEDataset(fakeFolder=self.fakeFolder, num_processes=4)
        self.model = CIFAKEClassifier()
        self.model = self.model.to(fetchDevice())

        # Hyperparameters
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.epochs = 5

        # Create the training and testing splits
        self.train_size = int(50_000)
        self.test_size = len(self.data_set) - self.train_size
        self.train_dataset, self.test_dataset = random_split(self.data_set, [self.train_size, self.test_size])


    def train(self):
        # Dataloader for batch training
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        #self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                labels = labels.float()  # BCELoss expects labels to be in float format

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs).squeeze()  # Remove unnecessary dimensions
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 mini-batches
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                    running_loss = 0.0

        torch.save(self.model.state_dict(), 'weights/' + self.fakeFolder + '_model.pth')
        print('Finished Training ' + self.fakeFolder)

    def loadWeight(self, weightPath):
        self.model.load_state_dict(torch.load(weightPath))

    def fetchTestDataSet(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    
    def test_model(self, datasetClassifier):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in datasetClassifier.fetchTestDataSet():
                images, labels = data
                outputs = self.model(images).squeeze()  # Remove unnecessary dimensions
                predicted = torch.round(outputs)  # Round to get binary predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
        
    def getName(self):
        return self.fakeFolder



fakeDatasets = list(os.listdir("STABLEDIFFUSION"))
datasetClassifiers = [0]*len(fakeDatasets)
counter = 0
for i in fakeDatasets:
    datasetClassifiers[counter] = DatasetClassifier("STABLEDIFFUSION/d21cifake-512pxjpg")
    datasetClassifiers[counter].train()

