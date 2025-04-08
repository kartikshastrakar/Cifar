import torch

# Get predictions from a model
def get_predictions(model, dataloader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = torch.softmax(model(inputs), dim=1)
            all_probs.append(outputs.cpu())
    return torch.cat(all_probs, dim=0)
