import torch
from utils import get_predictions
from data_preprocessing import testloader, test_labels
from train_resnet import train_resnet
from train_vgg import train_vgg

# Load models
resnet = train_resnet()
vgg = train_vgg()

# Get predictions
resnet_probs = get_predictions(resnet, testloader)
vgg_probs = get_predictions(vgg, testloader)

# Average probabilities
fused_probs = (resnet_probs + vgg_probs) / 2
fused_preds = torch.argmax(fused_probs, dim=1)

# Calculate accuracy
correct = (fused_preds == test_labels).sum().item()
accuracy = 100 * correct / len(test_labels)
print(f"Late Fusion Accuracy: {accuracy:.2f}%")
