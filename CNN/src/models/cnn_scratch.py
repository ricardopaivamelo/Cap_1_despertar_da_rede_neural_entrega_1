"""
CNN treinada do zero para classificação binária
Arquitetura simples e eficiente
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class SimpleCNN(nn.Module):
    """
    Rede Convolucional simples para classificação binária

    Arquitetura:
        - 3 blocos convolucionais (conv + relu + maxpool)
        - 2 camadas fully connected
        - Dropout para regularização
    """

    def __init__(self, num_classes=2, dropout=0.5):
        super(SimpleCNN, self).__init__()

        # Bloco Convolucional 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )

        # Bloco Convolucional 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )

        # Bloco Convolucional 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )

        # Bloco Convolucional 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        )

        # Camadas Fully Connected
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc(x)

        return x

    def get_num_parameters(self):
        """Retorna número total de parâmetros treináveis"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNTrainer:
    """
    Classe para treinar a CNN do zero
    """

    def __init__(self, model, device='cuda', learning_rate=0.001):
        """
        Args:
            model: Modelo CNN
            device: 'cuda' ou 'cpu'
            learning_rate: Taxa de aprendizado
        """
        self.model = model.to(device)
        self.device = device

        # Loss e Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Scheduler para ajustar learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # Histórico de treinamento
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader):
        """Treina por uma época"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Estatísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Atualizar progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Valida o modelo"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Estatísticas
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, epochs=20):
        """
        Treina o modelo

        Args:
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            epochs: Número de épocas

        Returns:
            Histórico de treinamento
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_parameters():,}\n")

        best_val_acc = 0.0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)

            # Treinar
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validar
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Ajustar learning rate
            self.scheduler.step(val_loss)

            # Imprimir resultados
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Salvar melhor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")

        print(f"\n{'='*40}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"{'='*40}\n")

        return self.history

    def predict(self, test_loader):
        """
        Faz predições no conjunto de teste

        Args:
            test_loader: DataLoader de teste

        Returns:
            y_true, y_pred (numpy arrays)
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_labels), np.array(all_preds)


def create_simple_cnn(num_classes=2, device='cuda'):
    """
    Cria e retorna uma CNN simples

    Args:
        num_classes: Número de classes
        device: 'cuda' ou 'cpu'

    Returns:
        Modelo CNN
    """
    model = SimpleCNN(num_classes=num_classes)

    # Verificar se CUDA está disponível
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'

    return model, device
