import torch

def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # Evaluation measures performance without calculating gradients.
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Weight by batch size to account for a smaller final batch.
            running_loss += loss.item() * labels.size(0)

    if total == 0:
        raise ValueError('Evaluation loader is empty.')

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy
