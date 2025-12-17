import torch
import torch.nn as nn
import torch.optim as optim
from sparseml.pytorch.optim import ScheduledModifierManager
import numpy as np
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3072, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1568, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, recipe, onnx_name):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    manager = ScheduledModifierManager.from_yaml(recipe)
    steps_per_epoch = len(train_dataset) // batch_size
    optimizer = manager.modify(model, optimizer, steps_per_epoch)

    criterion = nn.CrossEntropyLoss()

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

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Accuracy: {accuracy:.2f}%")

    manager.finalize(model)

    dummy_input = torch.randn(100, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        f"{onnx_name}.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    batch_size = 64
    epochs = 40
    learning_rate = 0.001

    (cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = \
        tf.keras.datasets.cifar10.load_data()

    cifar_train_labels, cifar_test_labels = cifar_train_labels.reshape(-1), cifar_test_labels.reshape(-1)

    cifar_train_images = np.transpose(cifar_train_images, (0, 3, 1, 2)).astype('float32') / 255
    cifar_test_images = np.transpose(cifar_test_images, (0, 3, 1, 2)).astype('float32') / 255

    val_size = int(0.2 * cifar_train_images.shape[0])
    train_images = cifar_train_images[:-val_size]
    train_labels = cifar_train_labels[:-val_size]
    val_images = cifar_train_images[-val_size:]
    val_labels = cifar_train_labels[-val_size:]

    X_train = torch.from_numpy(train_images).float()
    y_train = torch.from_numpy(train_labels).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val = torch.from_numpy(val_images).float()
    y_val = torch.from_numpy(val_labels).long()
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    X_test = torch.from_numpy(cifar_test_images).float()
    y_test = torch.from_numpy(cifar_test_labels).long()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_model = SimpleCNN().to(device)
    train_model(cnn_model, "cnn_recipe.yaml", "cnn_model_pruned")

    plain_model = SimpleNN().to(device)
    train_model(plain_model, "plain_recipe.yaml", "plain_model_pruned")

    # check deepsparse inference time

    # import timeit
    # from deepsparse import compile_model
    #
    # batch_size = 128
    # inputs = [np.random.randn(batch_size, 3, 32, 32).astype(np.float32)]
    # engine = compile_model("cnn_model_pruned", batch_size)
    # deepsparse_inference_time = timeit.timeit(lambda: engine(inputs), number=1000)
    # print(deepsparse_inference_time)
