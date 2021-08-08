import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
drop_prob = 0.2
weight_decay_lambda = 0.0001
learning_rate = 0.001

# 파이토치에서 제공하는 MNIST dataset
train_dev_dataset = torchvision.datasets.MNIST(root='./data',train=True, 
transform=transforms.ToTensor(), download=True)

train_dataset, dev_dataset = torch.utils.data.random_split(train_dev_dataset, [50000, 10000])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
transform=transforms.ToTensor())
# 배치 단위로 데이터를 처리해주는 Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
batch_size=batch_size, shuffle=True)

dev_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) #hidden connector 784X500
        torch.nn.init.xavier_normal_(self.fc1.weight) # weigth 는 사비에르로 초기화
        torch.nn.init.zeros_(self.fc1.bias) # 바이어스는 0으로 초기화
        self.fc2 = nn.Linear(hidden_size, hidden_size) #500x500
        torch.nn.init.xavier_normal_(self.fc2.weight) # 사비에르 방법으로~
        torch.nn.init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(hidden_size, num_classes) #500x10
        torch.nn.init.xavier_normal_(self.fc3.weight) # 사비에르 방법으로~
        torch.nn.init.zeros_(self.fc3.bias)
        
        self.dropout = nn.Dropout(drop_prob) #drop out : 일부러 네트워크 일부를 생략하는 것
        
    def forward(self, x):
        out = F.relu(self.fc1(x)) # 연산 (100, 784)*(500,500) 100은 배치 수
        out = self.dropout(out)
        out = F.relu(self.fc2(out)) #(100, 500)*(500,500)
        out = self.dropout(out)
        out = self.fc3(out) #(100, 500)*(500, 10)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_lambda)
# model.parameters -> 가중치 w들을 의미

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
def evaluation(data_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device) # 데이터를 장치에 올려줌
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

loss_arr = []
max = 0.0
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): 
        model.train()
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device) 
        # Forward pass
        outputs = model(images) #10개의 클래스에 대한 결과(실수값)
        loss = criterion(outputs, labels) #(오차 구함)
        # Backward and optimize
        optimizer.zero_grad() # iteration 마다 gradient를 0으로 초기화
        loss.backward() # 가중치 w에 대해 loss를 미분
        optimizer.step() # 가중치들을 업데이트
        if (i+1) % 100 == 0:
            loss_arr.append(loss)
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            with torch.no_grad():
                model.eval()
                acc = evaluation(dev_loader)
                if max < acc:
                    max = acc
                    print("max dev_accuracy : ", max)
                    torch.save(model.state_dict(), 'model.ckpt')
            

# Save the model checkpoint

with torch.no_grad():
    last_acc = evaluation(test_loader)
    print("Last Accuracy of the network on the 10000 test images: {} %".format(last_acc))
    
    torch.load('model.ckpt')
    best_acc = evaluation(test_loader)
    print("Best Accuracy of the network on the 1000 test images{} %".format(best_acc))
    
torch.save(model.state_dict(), './model.ckpt')
plt.plot(loss_arr)
plt.show()