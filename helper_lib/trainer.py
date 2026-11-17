from pathlib import Path

import torch
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .generator import EnergyReplayBuffer


def train_diffusion(
    model, data_loader, criterion, optimizer, device="cpu", epochs=10
):
    """Train on RGB images in [0, 1]; the model normalizes internally."""
    if epochs < 1:
        raise ValueError("epochs must be positive.")

    model.to(device)

    for epoch in range(epochs):
        model.train()
        model.ema_network.eval()
        total_loss = 0.0
        total_images = 0

        progress = tqdm(
            data_loader,
            ncols=120,
            desc=f"Diffusion epoch {epoch + 1}/{epochs}",
        )
        for images, _ in progress:
            images = images.to(device)
            loss = model.train_step(images, optimizer, criterion)

            total_loss += loss * images.size(0)
            total_images += images.size(0)
            progress.set_postfix(loss=loss)

        if total_images == 0:
            raise ValueError("Training loader is empty.")

        print(f"Epoch {epoch + 1}: Noise loss={total_loss / total_images:.4f}")

    return model


def train_energy(
    model, data_loader, optimizer, device="cpu", epochs=10, *,
    alpha=0.1, steps=60, step_size=10.0, noise_std=0.005,
):
    """Train on 32x32 RGB images normalized to [-1, 1]."""
    if epochs < 1 or steps < 1 or step_size <= 0:
        raise ValueError("epochs, steps and step_size must be positive.")
    if alpha < 0 or noise_std < 0:
        raise ValueError("alpha and noise_std cannot be negative.")

    model.to(device)
    buffer = EnergyReplayBuffer(model, device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_images = 0
        progress = tqdm(
            data_loader, ncols=120, desc=f"Energy epoch {epoch + 1}/{epochs}"
        )

        for real, _ in progress:
            real = real.to(device)
            batch_size = real.size(0)
            real = (real + torch.randn_like(real) * noise_std).clamp(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            fake = buffer.sample(batch_size, steps, step_size, noise_std)

            scores = model(torch.cat([real, fake], dim=0).detach())
            real_energy, fake_energy = scores.split(batch_size, dim=0)

            contrastive_loss = real_energy.mean() - fake_energy.mean()
            regularization = alpha * (
                real_energy.square().mean() + fake_energy.square().mean()
            )
            loss = contrastive_loss + regularization

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_images += batch_size
            progress.set_postfix(
                loss=loss.item(),
                real_energy=real_energy.mean().item(),
                fake_energy=fake_energy.mean().item(),
            )

        if total_images == 0:
            raise ValueError("Training loader is empty.")

        print(f"Epoch {epoch + 1}: Loss={total_loss / total_images:.4f}")

    return model


def train_gan(model, data_loader, device="cpu", epochs=10, lr=0.0002):
    if epochs < 1 or lr <= 0:
        raise ValueError("epochs and lr must be positive.")

    model.to(device)
    model.train()
    generator = model.generator
    discriminator = model.discriminator

    criterion = torch.nn.BCELoss()
    opt_gen = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(0.5, 0.999)
    )
    opt_disc = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
    )

    for epoch in range(epochs):
        batches = 0
        progress = tqdm(
            data_loader, ncols=120, desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for real, _ in progress:
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            noise = torch.randn(batch_size, generator.z_dim, device=device)
            fake = generator(noise)

            # Train the discriminator without updating the generator.
            opt_disc.zero_grad(set_to_none=True)
            loss_real = criterion(discriminator(real), real_labels)
            loss_fake = criterion(discriminator(fake.detach()), fake_labels)
            loss_disc = (loss_real + loss_fake) / 2
            loss_disc.backward()
            opt_disc.step()

            # Train the generator through the updated discriminator.
            opt_gen.zero_grad(set_to_none=True)
            opt_disc.zero_grad(set_to_none=True)
            discriminator.requires_grad_(False)
            try:
                loss_gen = criterion(discriminator(fake), real_labels)
                loss_gen.backward()
                opt_gen.step()
            finally:
                discriminator.requires_grad_(True)

            progress.set_postfix(
                D_loss=loss_disc.item(), G_loss=loss_gen.item()
            )
            batches += 1

        if batches == 0:
            raise ValueError("Training loader is empty.")

    return model


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
