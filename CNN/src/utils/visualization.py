"""
Funções para visualização de resultados
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import cv2


def plot_sample_images(image_paths, labels=None, n_cols=4, figsize=(15, 10)):
    """
    Plota uma grade de imagens de amostra

    Args:
        image_paths: Lista de caminhos das imagens
        labels: Lista de labels correspondentes (opcional)
        n_cols: Número de colunas na grade
        figsize: Tamanho da figura
    """
    n_images = len(image_paths)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]

    for idx, (img_path, ax) in enumerate(zip(image_paths, axes)):
        if idx < n_images:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)

            if labels is not None:
                ax.set_title(labels[idx], fontsize=10)
            else:
                ax.set_title(Path(img_path).name, fontsize=8)

            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Plota matriz de confusão

    Args:
        cm: Matriz de confusão
        class_names: Nomes das classes
        title: Título do gráfico
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_training_history(history, metrics=['loss', 'accuracy']):
    """
    Plota histórico de treinamento

    Args:
        history: Histórico de treinamento (dict ou History do Keras)
        metrics: Lista de métricas para plotar
    """
    if hasattr(history, 'history'):
        history = history.history

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_metric = history.get(metric, [])
        val_metric = history.get(f'val_{metric}', [])

        epochs = range(1, len(train_metric) + 1)

        ax.plot(epochs, train_metric, 'b-', label=f'Training {metric}')
        if val_metric:
            ax.plot(epochs, val_metric, 'r-', label=f'Validation {metric}')

        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison_bar(results_dict, metric='accuracy', title=None):
    """
    Plota gráfico de barras comparativo entre modelos

    Args:
        results_dict: Dicionário {modelo: valor}
        metric: Nome da métrica sendo comparada
        title: Título do gráfico
    """
    models = list(results_dict.keys())
    values = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(models, values, color=colors[:len(models)])

    # Adicionar valores em cima das barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel(metric.capitalize())
    ax.set_title(title or f'{metric.capitalize()} Comparison')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_time_comparison(results_dict, title='Time Comparison'):
    """
    Plota comparação de tempos (treino e inferência)

    Args:
        results_dict: Dict {modelo: {'train': X, 'inference': Y}}
        title: Título do gráfico
    """
    models = list(results_dict.keys())
    train_times = [results_dict[m].get('train', 0) for m in models]
    inference_times = [results_dict[m].get('inference', 0) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training time
    ax1.bar(models, train_times, color='#1f77b4')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time')
    ax1.grid(True, axis='y', alpha=0.3)

    for i, v in enumerate(train_times):
        ax1.text(i, v, f'{v:.2f}s', ha='center', va='bottom')

    # Inference time
    ax2.bar(models, inference_times, color='#ff7f0e')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Inference Time (per image)')
    ax2.grid(True, axis='y', alpha=0.3)

    for i, v in enumerate(inference_times):
        ax2.text(i, v, f'{v:.2f}ms', ha='center', va='bottom')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
