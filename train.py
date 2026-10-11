import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model


def main():
    torch.manual_seed(42)

    # Load data and reserve 10% of the training set for validation.
    # train=False selects the test set, not a separate validation set.
    dataset = get_data_loader("data", batch_size=64).dataset
    train_data, val_data = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = get_data_loader("data", batch_size=64, train=False)

    # Initialize model and training components.
    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train from scratch with checkpoint saving.
    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=10, checkpoint_dir="checkpoints/assignment2",
    )

    # Evaluate the final model on the test set.
    avg_loss, accuracy = evaluate_model(trained_model, test_loader, criterion)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
