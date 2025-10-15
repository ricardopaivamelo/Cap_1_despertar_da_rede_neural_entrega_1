"""
Wrapper para modelos YOLO (tradicional e custom)
"""

import time
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
import torch


class YOLOPredictor:
    """
    Classe wrapper para facilitar uso do YOLO
    """

    def __init__(self, model_path='yolov8n.pt', device='cuda'):
        """
        Args:
            model_path: Caminho do modelo YOLO (.pt)
            device: 'cuda', 'cpu', ou número da GPU
        """
        self.device = device if torch.cuda.is_available() else 'cpu'

        if self.device == 'cpu':
            print("⚠️  CUDA not available, using CPU")

        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        print(f"Model loaded successfully on {self.device}")

    def predict_image(self, image_path, conf=0.25, save=False, save_dir='results'):
        """
        Faz predição em uma imagem

        Args:
            image_path: Caminho da imagem
            conf: Threshold de confiança
            save: Se True, salva imagem com detecções
            save_dir: Diretório para salvar resultados

        Returns:
            Resultados da predição
        """
        results = self.model.predict(
            image_path,
            conf=conf,
            save=save,
            project=save_dir,
            device=self.device
        )

        return results[0]  # Retorna primeiro resultado

    def predict_batch(self, image_paths, conf=0.25):
        """
        Faz predições em múltiplas imagens

        Args:
            image_paths: Lista de caminhos de imagens
            conf: Threshold de confiança

        Returns:
            Lista de resultados
        """
        results = self.model.predict(
            image_paths,
            conf=conf,
            device=self.device
        )

        return results

    def predict_folder(self, folder_path, conf=0.25):
        """
        Faz predições em todas as imagens de uma pasta

        Args:
            folder_path: Caminho da pasta
            conf: Threshold de confiança

        Returns:
            Dict com resultados por imagem
        """
        folder = Path(folder_path)
        image_paths = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))

        print(f"Found {len(image_paths)} images in {folder_path}")

        results_dict = {}
        for img_path in image_paths:
            result = self.predict_image(str(img_path), conf=conf)
            results_dict[img_path.name] = result

        return results_dict

    def measure_inference_time(self, image_path, n_runs=10):
        """
        Mede tempo médio de inferência

        Args:
            image_path: Caminho da imagem de teste
            n_runs: Número de execuções para média

        Returns:
            Tempo médio em milissegundos
        """
        # Warm-up
        _ = self.predict_image(image_path)

        # Medir tempo
        times = []
        for _ in range(n_runs):
            start = time.time()
            _ = self.predict_image(image_path)
            elapsed = (time.time() - start) * 1000  # converter para ms
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"Inference time: {avg_time:.2f} ± {std_time:.2f} ms")

        return avg_time

    def get_model_info(self):
        """Retorna informações sobre o modelo"""
        return {
            'type': self.model.type,
            'task': self.model.task,
            'device': str(self.device),
            'names': self.model.names
        }


class YOLOTrainer:
    """
    Classe para treinar YOLO customizado
    """

    def __init__(self, model_name='yolov8n.pt'):
        """
        Args:
            model_name: Nome do modelo base (ex: 'yolov8n.pt', 'yolov5s.pt')
        """
        print(f"Initializing YOLO model: {model_name}")
        self.model = YOLO(model_name)

    def train(self, data_yaml, epochs=30, imgsz=640, batch=16,
              project='runs/train', name='custom_yolo'):
        """
        Treina o modelo YOLO

        Args:
            data_yaml: Caminho do arquivo de configuração do dataset
            epochs: Número de épocas
            imgsz: Tamanho da imagem
            batch: Tamanho do batch
            project: Diretório do projeto
            name: Nome do experimento

        Returns:
            Resultados do treinamento
        """
        print(f"\nStarting YOLO training...")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Batch size: {batch}\n")

        start_time = time.time()

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            patience=10,  # Early stopping
            save=True,
            plots=True
        )

        training_time = time.time() - start_time

        print(f"\nTraining completed in {training_time:.2f} seconds")

        return results, training_time

    def validate(self, data_yaml):
        """
        Valida o modelo treinado

        Args:
            data_yaml: Caminho do arquivo de configuração

        Returns:
            Métricas de validação
        """
        print("\nValidating model...")
        metrics = self.model.val(data=data_yaml)

        return metrics


def compare_yolo_models(traditional_model_path, custom_model_path,
                        test_images_path, conf=0.25):
    """
    Compara YOLO tradicional vs YOLO custom

    Args:
        traditional_model_path: Caminho do modelo tradicional
        custom_model_path: Caminho do modelo customizado
        test_images_path: Caminho das imagens de teste
        conf: Threshold de confiança

    Returns:
        Dict com comparação de resultados
    """
    print("\n" + "="*60)
    print("COMPARING YOLO MODELS")
    print("="*60 + "\n")

    # Modelo tradicional
    print("1. Loading traditional YOLO...")
    yolo_trad = YOLOPredictor(traditional_model_path)

    # Modelo custom
    print("\n2. Loading custom YOLO...")
    yolo_custom = YOLOPredictor(custom_model_path)

    # Testar em imagens
    test_folder = Path(test_images_path)
    test_images = list(test_folder.glob('*.jpg'))

    if not test_images:
        print("⚠️  No test images found!")
        return None

    # Medir tempo de inferência
    print("\n3. Measuring inference time...")
    test_img = str(test_images[0])

    trad_time = yolo_trad.measure_inference_time(test_img)
    custom_time = yolo_custom.measure_inference_time(test_img)

    # Resultados
    comparison = {
        'traditional': {
            'model': traditional_model_path,
            'inference_time_ms': trad_time,
        },
        'custom': {
            'model': custom_model_path,
            'inference_time_ms': custom_time,
        }
    }

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Traditional YOLO: {trad_time:.2f} ms")
    print(f"Custom YOLO: {custom_time:.2f} ms")
    print("="*60 + "\n")

    return comparison
