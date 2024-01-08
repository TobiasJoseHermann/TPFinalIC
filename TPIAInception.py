import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import random_split


class PotatoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Potato_Early_Blight', 'Potato_Healthy', 'Potato_Late_Blight']
        self.images = []
        self.labels = []

        for class_id, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, image_name))
                self.labels.append(class_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 16

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print(device)


# Dividir en train y test
root_dir_test = '.\Potato Test'
# root_dir_train = '.\Potato Train Augmentation'
root_dir_train = '.\Potato Train'
root_dir_val = '.\Potato Val'

dataset_test = PotatoDataset(root_dir=root_dir_test, transform=transform)
dataset_train = PotatoDataset(root_dir=root_dir_train, transform=transform)
dataset_val = PotatoDataset(root_dir=root_dir_val, transform=transform)

# Define the size of the test set

# Split the dataset

# Create data loaders for the training and test sets
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)


classes = ('Potato_Early_Blight', 'Potato_Healthy', 'Potato_Late_Blight')


import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

################################
# Importar el modelo Inception #
################################


from torchvision import models

# Load the pretrained Inception model
net = models.inception_v3(pretrained=True)

# Freeze the model parameters (so we don't backprop through them)
# for param in net.parameters():
#     param.requires_grad = False

# Replace the final layer with a new one with 3 outputs
num_ftrs = net.fc.in_features
net.fc = torch.nn.Linear(num_ftrs, 3)

# Move the model to the GPU if available
model = net.to(device)


import torch.optim as optim

#################
# Entrenamiento #
#################

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_train_vector = []
loss_val_vector = []
accuracy_vector = []
accuracy_balanceado_vector = []

patience = 5
best_loss = float('inf')
epochs_no_improve = 0

best_accuracy = 0

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    epoch_loss = 0.0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device) # send to GPU

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, _ = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        epoch_loss += loss.item()

        correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

        if i%6 == 5:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 6:.3f}')
            running_loss = 0.0
    
    # Es len(trainloader) porque es el número de batches
    loss_train_vector.append(epoch_loss/len(trainloader))
    accuracy_vector.append(correct/len(trainloader))



    # Accuracy balanceado validation
    
    correct = 0
    correct_late=0
    correct_early=0
    correct_heal=0
    total = 0
    total_late = 0
    total_early = 0
    total_heal = 0


    # Validation
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs,_ = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()



            # Accuracy balanceado validation
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_early += (labels==0).sum().item()
            total_heal += (labels==1).sum().item()
            total_late += (labels==2).sum().item()
            correct += (predicted == labels).sum().item()
            correct_early += ((predicted == labels) & (labels == 0)).sum().item()
            correct_heal += ((predicted == labels) & (labels == 1)).sum().item()
            correct_late += ((predicted == labels) & (labels == 2)).sum().item()


    val_loss /= len(valloader)
    print(val_loss)


    # Accuracy balanceado validation
    accuracy_late = correct_late/total_late
    accuracy_early = correct_early/total_early
    accuracy_heal = correct_heal/total_heal
    accuracy_balanceado = (accuracy_late+accuracy_early+accuracy_heal) / 3
    accuracy_balanceado_vector.append(accuracy_balanceado)
    print('accuracy balanceado: ', accuracy_balanceado*100)

    # Check if validation loss has improved
    # if val_loss < best_loss:
    #     best_loss = val_loss
    #     epochs_no_improve = 0
    #     #TODO: Cargar best model state desde memoria
    #     torch.save(net.state_dict(), 'best_model.pth')
    if accuracy_balanceado > best_accuracy:
        best_accuracy = accuracy_balanceado
        epochs_no_improve = 0
        best_model_state = net.state_dict()
        torch.save(net.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping!')
            break
    
    loss_val_vector.append(val_loss)


# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# Carga el estado del mejor modelo
# net.load_state_dict(torch.load('best_model.pth'))
net.load_state_dict(best_model_state)

# Cambia el modelo a modo de evaluación
net.eval()

print('Finished Training')


############
# Graficar #
############

epochs = range(1, len(loss_train_vector) + 1)
# Crear el gráfico de línea
plt.figure(1)
plt.plot(loss_train_vector)
# Añadir etiquetas a los ejes
plt.xlabel('Epocas')
plt.ylabel('Loss')
# Añadir un título al gráfico
plt.title('Gráfico de Loss')

# plt.plot(accuracy_balanceado_vector)

plt.figure(2)
plt.plot(accuracy_vector)
# Añadir etiquetas a los ejes
plt.xlabel('Epocas')
plt.ylabel('Accuracy')
# Añadir un título al gráfico
plt.title('Gráfico de Accuracy')

# Mostrar el gráfico
plt.show()


###################
# Funcion de test #
###################

correct = 0
correct_late=0
correct_early=0
correct_heal=0
total = 0
total_late = 0
total_early = 0
total_heal = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device) # send to GPU
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        total_early += (labels==0).sum().item()
        total_heal += (labels==1).sum().item()
        total_late += (labels==2).sum().item()
        correct += (predicted == labels).sum().item()
        correct_early += ((predicted == labels) & (labels == 0)).sum().item()
        correct_heal += ((predicted == labels) & (labels == 1)).sum().item()
        correct_late += ((predicted == labels) & (labels == 2)).sum().item()
        

accuracy_late = correct_late/total_late
accuracy_early = correct_early/total_early
accuracy_heal = correct_heal/total_heal

print("Accuracy early: ",accuracy_early)
print("Accuracy late: ",accuracy_late)
print("Accuracy heal: ",accuracy_heal)

print(f'Accuracy sin balanciar: {100 * correct // total} %')
print(f'Accuracy balanciado: {100*(accuracy_late+accuracy_early+accuracy_heal) //  3} %')




################################
# Guardar resultados en un txt #
################################

txtString = f"Model: {net}\nBatch size: {batch_size}\nEpochs: {epoch}\nOptimizer: {optimizer}\nCriterion: {criterion}\nDevice: {device}\nAccuracy: {100 * correct // total} %\nRootDir: {root_dir_train}\n\n"

txtString += f'Accuracy early: {accuracy_early:.5f} %\n'
txtString += f'Accuracy late: {accuracy_late:.5f} %\n'
txtString += f'Accuracy heal: {accuracy_heal:.5f} %\n'

import datetime
now = datetime.datetime.now()
filename = now.strftime("%Y-%m-%d %H_%M_%S") + ".txt"
with open(f'Resultados/{filename}', 'w') as file:
    file.write(txtString)





