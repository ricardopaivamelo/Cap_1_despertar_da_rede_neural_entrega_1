# Como Usar

## 1. Setup R�pido

```bash
source .venv/bin/activate
python run_experiments.py  # 3 minutos
```

## 2. Notebooks

```bash
jupyter notebook
```

Execute em ordem:
1. `00_quick_test.ipynb` - Teste r�pido (5min)
2. `01_yolo_tradicional.ipynb` - YOLO pr�-treinado (10min)
3. `02_yolo_custom.ipynb` - YOLO custom (30-40min) �
4. `03_cnn_scratch.ipynb` - CNN do zero (20min)
5. `04_comparacao.ipynb` - An�lise final (5min)

## 3. V�deo

Grave 5min mostrando:
- Notebooks executados
- Tabela comparativa
- Sua conclus�o

## Resultados

| Modelo | Precis�o | Velocidade |
|--------|----------|------------|
| YOLO Trad | ~65% | 205ms |
| YOLO Custom | ~85-90% | 25-30ms |
| **CNN** | **100%** | **1.25ms** � |

CNN � 164x mais r�pida!

## Arquivos

- `README.md` - Vis�o geral
- `RESULTADOS_FINAIS.md` - Resultados
- `experiment_results.log` - Log do treino
- `results/` - M�tricas
- `models/` - Modelos salvos
