"""
Carregamento e preprocessamento de datasets
"""

import os
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CoffeeMountainsDataset(Dataset):
    """
    Dataset customizado para classificação Coffee vs Mountains

    Classes:
        0: Coffee
        1: Mountains
    """

    def __init__(self, root_dir, split='train', transform=None, img_size=(224, 224)):
        """
        Args:
            root_dir: Diretório raiz do dataset
            split: 'train', 'val', ou 'test'
            transform: Transformações a serem aplicadas
            img_size: Tamanho para redimensionar imagens (H, W)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size

        # Mapeamento de split para nome das pastas
        split_folders = {
            'train': 'Treino',
            'val': 'Validacao',
            'test': 'Teste'
        }

        # Caminhos das classes
        self.classes = ['coffee', 'mountains']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Carregar imagens e labels
        self.samples = []
        self._load_samples(split_folders[split])

    def _load_samples(self, folder_name):
        """Carrega amostras do dataset"""
        # Coffee images
        coffee_dir = self.root_dir / 'Coffee' / folder_name
        if coffee_dir.exists():
            for img_path in coffee_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 0))  # 0 = coffee

        # Mountains images
        mountains_dir = self.root_dir / 'Mountains' / folder_name
        if mountains_dir.exists():
            for img_path in mountains_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 1))  # 1 = mountains

        print(f"Loaded {len(self.samples)} images for {self.split} split")
        print(f"  - Coffee: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  - Mountains: {sum(1 for _, label in self.samples if label == 1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Retorna uma amostra do dataset"""
        img_path, label = self.samples[idx]

        # Carregar imagem
        image = Image.open(img_path).convert('RGB')

        # Aplicar transformações
        if self.transform:
            image = self.transform(image)
        else:
            # Transformação padrão
            image = transforms.Resize(self.img_size)(image)
            image = transforms.ToTensor()(image)

        return image, label

    def get_class_name(self, idx):
        """Retorna nome da classe dado o índice"""
        return self.classes[idx]


def get_default_transforms(img_size=(224, 224), augment=True):
    """
    Retorna transformações padrão para o dataset

    Args:
        img_size: Tamanho da imagem (H, W)
        augment: Se True, aplica data augmentation

    Returns:
        Dict com transformações para train e val/test
    """
    # Normalização padrão do ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Transformações de treino (com augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize
    ]) if augment else transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])

    # Transformações de validação/teste (sem augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])

    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def create_dataloaders(root_dir, batch_size=32, img_size=(224, 224),
                       augment=True, num_workers=2):
    """
    Cria DataLoaders para train, val e test

    Args:
        root_dir: Diretório raiz do dataset
        batch_size: Tamanho do batch
        img_size: Tamanho da imagem (H, W)
        augment: Se True, aplica data augmentation no treino
        num_workers: Número de workers para carregar dados

    Returns:
        Dict com DataLoaders {'train': ..., 'val': ..., 'test': ...}
    """
    transforms_dict = get_default_transforms(img_size, augment)

    # Criar datasets
    train_dataset = CoffeeMountainsDataset(
        root_dir, split='train', transform=transforms_dict['train']
    )
    val_dataset = CoffeeMountainsDataset(
        root_dir, split='val', transform=transforms_dict['val']
    )
    test_dataset = CoffeeMountainsDataset(
        root_dir, split='test', transform=transforms_dict['test']
    )

    # Criar dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        ),
        'val': DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        ),
        'test': DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )
    }

    return dataloaders, {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}


def load_images_from_folder(folder_path, img_size=(224, 224)):
    """
    Carrega todas as imagens de uma pasta

    Args:
        folder_path: Caminho da pasta
        img_size: Tamanho para redimensionar

    Returns:
        List de arrays numpy com imagens
    """
    images = []
    image_paths = []

    folder = Path(folder_path)
    for img_path in sorted(folder.glob('*.jpg')):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        images.append(img)
        image_paths.append(str(img_path))

    return np.array(images), image_paths
