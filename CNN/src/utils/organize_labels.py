"""
Script para organizar labels YOLO nas pastas corretas
"""

import os
import shutil
from pathlib import Path


def organize_yolo_labels(root_dir, labels_dir='Labels'):
    """
    Organiza labels YOLO nas pastas correspondentes às imagens

    Estrutura esperada:
        Coffee/
            Treino/ (imagens)
            Validacao/ (imagens)
            Teste/ (imagens)
        Mountains/
            Treino/ (imagens)
            Validacao/ (imagens)
            Teste/ (imagens)
        Labels/ (todos os .txt)

    Labels serão copiados para:
        Coffee/Treino/labels/
        Coffee/Validacao/labels/
        etc.

    Args:
        root_dir: Diretório raiz do projeto
        labels_dir: Nome da pasta com os labels
    """
    root_path = Path(root_dir)
    labels_path = root_path / labels_dir

    if not labels_path.exists():
        print(f"⚠️  Labels directory not found: {labels_path}")
        return

    print(f"Organizing labels from {labels_path}")
    print("=" * 60)

    # Pastas de classes e splits
    classes = ['Coffee', 'Mountains']
    splits = ['Treino', 'Validacao', 'Teste']

    organized_count = 0
    not_found_count = 0

    for class_name in classes:
        for split in splits:
            # Diretório de imagens
            img_dir = root_path / class_name / split

            if not img_dir.exists():
                print(f"⚠️  Directory not found: {img_dir}")
                continue

            # Criar diretório para labels
            label_dir = img_dir / 'labels'
            label_dir.mkdir(exist_ok=True)

            # Procurar labels correspondentes às imagens
            image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

            for img_file in image_files:
                # Nome do label correspondente
                label_name = img_file.stem + '.txt'
                label_source = labels_path / label_name
                label_dest = label_dir / label_name

                # Copiar label se existir
                if label_source.exists():
                    shutil.copy2(label_source, label_dest)
                    organized_count += 1
                else:
                    not_found_count += 1
                    print(f"⚠️  Label not found: {label_name}")

            print(f"✓ {class_name}/{split}: {len(list(label_dir.glob('*.txt')))} labels")

    print("=" * 60)
    print(f"Summary:")
    print(f"  - Labels organized: {organized_count}")
    print(f"  - Labels not found: {not_found_count}")
    print("=" * 60)


def verify_yolo_format(label_file):
    """
    Verifica se o arquivo de label está no formato YOLO correto

    Formato YOLO: class_id x_center y_center width height (normalized)

    Args:
        label_file: Caminho do arquivo de label

    Returns:
        True se válido, False caso contrário
    """
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                return False

            # Verificar se todos são números
            class_id, x, y, w, h = map(float, parts)

            # Verificar se valores estão normalizados (0-1)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                return False

        return True

    except Exception as e:
        print(f"Error reading {label_file}: {e}")
        return False


def check_dataset_structure(root_dir):
    """
    Verifica estrutura do dataset YOLO

    Args:
        root_dir: Diretório raiz
    """
    root_path = Path(root_dir)

    print("\n" + "=" * 60)
    print("DATASET STRUCTURE CHECK")
    print("=" * 60 + "\n")

    classes = ['Coffee', 'Mountains']
    splits = ['Treino', 'Validacao', 'Teste']

    total_images = 0
    total_labels = 0

    for class_name in classes:
        print(f"{class_name}:")
        for split in splits:
            img_dir = root_path / class_name / split
            label_dir = img_dir / 'labels'

            if img_dir.exists():
                images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                labels = list(label_dir.glob('*.txt')) if label_dir.exists() else []

                total_images += len(images)
                total_labels += len(labels)

                status = "✓" if len(images) == len(labels) else "⚠️"
                print(f"  {split}: {len(images)} images, {len(labels)} labels {status}")
            else:
                print(f"  {split}: Not found ⚠️")

        print()

    print("=" * 60)
    print(f"Total: {total_images} images, {total_labels} labels")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # Executar organização
    root_dir = '/home/ubuntu/Developer/Others/fiap-fase-06-cap-01'

    print("Starting label organization...\n")
    organize_yolo_labels(root_dir)

    print("\nChecking dataset structure...")
    check_dataset_structure(root_dir)

    print("\n✓ Done!")
