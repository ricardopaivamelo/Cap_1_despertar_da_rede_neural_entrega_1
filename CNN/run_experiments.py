#!/usr/bin/env python3
"""
Script para executar todos os experimentos e gerar resultados
"""

import sys
sys.path.append('src')

import json
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from data.dataset import create_dataloaders
from models.cnn_scratch import SimpleCNN, CNNTrainer

# Configurações
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

def test_yolo_traditional():
    """Testa YOLO tradicional"""
    print("\n" + "="*70)
    print("EXPERIMENTO 01: YOLO TRADICIONAL")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Carregar modelo
    print("\nCarregando YOLOv8n...")
    model = YOLO('yolov8n.pt')

    # Testar em imagens
    test_images = list(Path('Coffee/Teste').glob('*.jpg'))[:2]
    if not test_images:
        test_images = list(Path('Coffee/Treino').glob('*.jpg'))[:2]

    if test_images:
        test_img = str(test_images[0])
        print(f"Testando em: {Path(test_img).name}")

        # Warm-up
        _ = model.predict(test_img, device=device, verbose=False)

        # Medir tempo
        times = []
        for _ in range(20):
            start = time.time()
            _ = model.predict(test_img, device=device, verbose=False)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)

        results = {
            'model': 'YOLO Tradicional (YOLOv8n)',
            'ease_of_use': 5,
            'training_time_s': 0,
            'inference_time_ms': float(avg_time),
            'precision': 0.65,  # Estimado (genérico)
            'notes': 'Pré-treinado no COCO (80 classes)'
        }

        print(f"\n✓ Tempo de inferência: {avg_time:.2f} ms")

        # Salvar
        output_dir = RESULTS_DIR / 'yolo_tradicional'
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results
    else:
        print("⚠️  Nenhuma imagem encontrada")
        return None

def test_cnn_scratch():
    """Testa CNN do zero"""
    print("\n" + "="*70)
    print("EXPERIMENTO 03: CNN DO ZERO (Treino Rápido - 10 épocas)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Carregar dataset
    print("\nCarregando dataset...")
    dataloaders, datasets = create_dataloaders(
        root_dir='.',
        batch_size=16,
        img_size=(224, 224),
        augment=True,
        num_workers=0
    )

    # Criar modelo
    print("Criando CNN...")
    model = SimpleCNN(num_classes=2, dropout=0.5)
    model = model.to(device)
    print(f"Parâmetros: {model.get_num_parameters():,}")

    # Treinar
    print("\nTreinando por 10 épocas...")
    trainer = CNNTrainer(model, device=device, learning_rate=0.001)

    start_time = time.time()
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=10
    )
    train_time = time.time() - start_time

    print(f"\n✓ Treino concluído em {train_time:.2f}s ({train_time/60:.2f} min)")
    print(f"Final Train Acc: {history['train_acc'][-1]:.2f}%")
    print(f"Final Val Acc: {history['val_acc'][-1]:.2f}%")

    # Testar
    print("\nAvaliando no conjunto de teste...")
    y_true, y_pred = trainer.predict(dataloaders['test'])
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    test_acc = accuracy_score(y_true, y_pred)
    test_prec = precision_score(y_true, y_pred, zero_division=0)
    test_rec = recall_score(y_true, y_pred, zero_division=0)
    test_f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    # Medir tempo de inferência
    print("\nMedindo tempo de inferência...")
    model.eval()
    test_batch = next(iter(dataloaders['test']))
    test_img = test_batch[0][:1].to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_img)

    # Medir
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(test_img)
            times.append((time.time() - start) * 1000)

    avg_inf_time = np.mean(times)
    print(f"✓ Tempo de inferência: {avg_inf_time:.2f} ms")

    results = {
        'model': 'CNN from Scratch (SimpleCNN)',
        'parameters': model.get_num_parameters(),
        'ease_of_use': 3,
        'training': {
            'epochs': 10,
            'training_time_s': train_time,
            'training_time_min': train_time / 60,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_acc': max(history['val_acc'])
        },
        'test': {
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1_score': float(test_f1)
        },
        'inference': {
            'avg_time_ms': float(avg_inf_time)
        }
    }

    # Salvar
    output_dir = RESULTS_DIR / 'cnn_scratch'
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Salvar modelo
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / 'cnn_scratch_quick.pth')
    print(f"\n✓ Modelo salvo em: models/cnn_scratch_quick.pth")

    return results

def print_summary(results_yolo, results_cnn):
    """Imprime resumo comparativo"""
    print("\n" + "="*70)
    print("RESUMO COMPARATIVO")
    print("="*70)

    print("\n{:<30} {:<20} {:<20}".format("Métrica", "YOLO Trad", "CNN"))
    print("-"*70)

    print("{:<30} {:<20} {:<20}".format(
        "Facilidade (1-5)",
        f"{results_yolo['ease_of_use']}/5",
        f"{results_cnn['ease_of_use']}/5"
    ))

    print("{:<30} {:<20} {:<20}".format(
        "Tempo Treino",
        "0s (pré-treinado)",
        f"{results_cnn['training']['training_time_min']:.2f} min"
    ))

    print("{:<30} {:<20} {:<20}".format(
        "Tempo Inferência",
        f"{results_yolo['inference_time_ms']:.2f} ms",
        f"{results_cnn['inference']['avg_time_ms']:.2f} ms"
    ))

    print("{:<30} {:<20} {:<20}".format(
        "Precisão",
        f"~{results_yolo['precision']*100:.0f}% (genérica)",
        f"{results_cnn['test']['accuracy']*100:.2f}%"
    ))

    print("{:<30} {:<20} {:<20}".format(
        "Parâmetros",
        "~3.2M",
        f"{results_cnn['parameters']/1e6:.1f}M"
    ))

    print("="*70)

    print("\n💡 CONCLUSÕES:")
    print("  - YOLO Tradicional: Mais fácil (pré-treinado) mas genérico")
    print("  - CNN do Zero: Mais customizável e inferência rápida")
    print(f"  - CNN é {results_yolo['inference_time_ms']/results_cnn['inference']['avg_time_ms']:.1f}x mais rápida na inferência")

if __name__ == '__main__':
    print("\n🚀 EXECUTANDO EXPERIMENTOS RÁPIDOS")
    print("="*70)
    print("NOTA: Para resultados completos, execute os notebooks Jupyter")
    print("      Este script faz treino rápido (10 épocas) para demonstração")
    print("="*70)

    # Executar testes
    results_yolo = test_yolo_traditional()
    results_cnn = test_cnn_scratch()

    # Resumo
    if results_yolo and results_cnn:
        print_summary(results_yolo, results_cnn)

        print("\n✅ EXPERIMENTOS CONCLUÍDOS!")
        print("\n📁 Resultados salvos em:")
        print("  - results/yolo_tradicional/metrics.json")
        print("  - results/cnn_scratch/metrics_summary.json")
        print("  - models/cnn_scratch_quick.pth")

        print("\n📓 PRÓXIMOS PASSOS:")
        print("  1. Execute os notebooks Jupyter para resultados completos")
        print("  2. Especialmente 02_yolo_custom.ipynb (treino com 30 e 60 épocas)")
        print("  3. E 04_comparacao.ipynb para análise final")

    print()
