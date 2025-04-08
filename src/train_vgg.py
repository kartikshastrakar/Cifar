import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_preprocessing import trainloader, device

# Define and Train VGG Model
def train_vgg():
    vgg = models.vgg16(pretrained=True)
    for param in vgg.features.parameters():
        param.requires_grad = False
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 10)
    vgg = vgg.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(3):
        vgg.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = vgg(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

    torch.save(vgg.state_dict(), './models/vgg16.pth')
    print("VGG training complete.")
    return vgg
