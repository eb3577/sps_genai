from pathlib import Path

import torch
from tqdm import tqdm

from .checkpoints import save_checkpoint


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints",
):

    datalogs = []
    best_accuracy = float("-inf")

    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        # Training
        model.train()
        train_loader_with_progress = tqdm(
            iterable=train_loader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for batch_number, (inputs, labels) in enumerate(
            train_loader_with_progress
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track training metrics.
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix({
                    "avg accuracy": (
                        f"{running_correct / running_total:.3f}"
                    ),
                    "avg loss": f"{running_loss / running_total:.4f}",
                })

                datalogs.append({
                    "epoch": epoch + (batch_number + 1) / len(train_loader),
                    "train_loss": running_loss / running_total,
                    "train_accuracy": running_correct / running_total,
                })


        if running_total == 0:
            raise ValueError("Training loader is empty.")

        epoch_loss = running_loss / running_total
        epoch_accuracy = 100 * running_correct / running_total


        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        model.eval()
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                _, val_predicted = torch.max(val_outputs, 1)

                val_total += val_labels.size(0)
                val_correct += (
                    val_predicted == val_labels
                ).sum().item()
                val_running_loss += loss.item() * val_labels.size(0)

        if val_total == 0:
            raise ValueError("Validation loader is empty.")

        val_loss = val_running_loss / val_total
        val_accuracy = 100 * val_correct / val_total

        datalogs.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": running_correct / running_total,
            "val_loss": val_loss,
            "val_accuracy": val_correct / val_total,
        })


        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_loss,
            val_accuracy,
            checkpoint_dir=checkpoint_dir,
        )


        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_path = save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_accuracy,
                checkpoint_dir=Path(checkpoint_dir) / "best",
            )
            print(
                f"New best model saved! "
                f"Validation accuracy: {val_accuracy:.2f}%"
            )
            print(f"Best checkpoint saved: {best_path}")

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%"
        )
        print(
            f"Validation: "
            f"Loss={val_loss:.4f}, Accuracy={val_accuracy:.2f}%"
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Finished Training")
    return model