import torch
import torch.nn as nn
import torch.optim as optim

from model import build_model
from dataset import get_dataloaders

def train_model(data_dir, num_classes, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

    train_loader, val_loader = get_dataloaders(data_dir)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f}")

if __name__ == "__main__":
    train_model("data/", num_classes=10)
