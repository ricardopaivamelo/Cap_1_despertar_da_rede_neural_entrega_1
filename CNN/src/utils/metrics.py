"""
Funções para cálculo de métricas de avaliação
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


class Timer:
    """Context manager para medir tempo de execução"""

    def __init__(self):
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_time = time.time() - self.start_time

    def get_elapsed_time(self):
        """Retorna tempo decorrido em segundos"""
        return self.elapsed_time


def calculate_metrics(y_true, y_pred, average='binary'):
    """
    Calcula métricas de classificação

    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        average: Tipo de média ('binary', 'macro', 'weighted')

    Returns:
        Dict com métricas calculadas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Calcula matriz de confusão

    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo

    Returns:
        Matriz de confusão (numpy array)
    """
    return confusion_matrix(y_true, y_pred)


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Imprime relatório de classificação detalhado

    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        class_names: Nomes das classes
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))


def compare_models(results_dict):
    """
    Compara resultados de múltiplos modelos

    Args:
        results_dict: Dict {modelo_nome: {metric: value}}

    Returns:
        String formatada com comparação
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    # Header
    models = list(results_dict.keys())
    print(f"{'Metric':<20}", end="")
    for model in models:
        print(f"{model:<20}", end="")
    print()
    print("-" * 80)

    # Get all metrics
    all_metrics = set()
    for model_results in results_dict.values():
        all_metrics.update(model_results.keys())

    # Print each metric
    for metric in sorted(all_metrics):
        print(f"{metric:<20}", end="")
        for model in models:
            value = results_dict[model].get(metric, 'N/A')
            if isinstance(value, (int, float)):
                print(f"{value:<20.4f}", end="")
            else:
                print(f"{value:<20}", end="")
        print()

    print("="*80)


def calculate_inference_time(model, data_loader, n_samples=None):
    """
    Calcula tempo médio de inferência

    Args:
        model: Modelo treinado
        data_loader: DataLoader com dados de teste
        n_samples: Número de amostras para testar (None = todas)

    Returns:
        Tempo médio em milissegundos por imagem
    """
    times = []
    count = 0

    for batch in data_loader:
        if n_samples and count >= n_samples:
            break

        with Timer() as timer:
            _ = model(batch)

        times.append(timer.get_elapsed_time())
        count += len(batch) if hasattr(batch, '__len__') else 1

    avg_time_ms = (np.mean(times) * 1000) / (count / len(times))
    return avg_time_ms
