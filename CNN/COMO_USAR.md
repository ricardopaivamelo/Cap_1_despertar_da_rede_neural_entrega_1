# Como Usar

## 1. Setup Rápido

```bash
source .venv/bin/activate
python run_experiments.py  # 3 minutos
```

## 2. Notebooks

```bash
jupyter notebook
```

Execute em ordem:
1. `00_quick_test.ipynb` - Teste rápido (5min)
2. `01_yolo_tradicional.ipynb` - YOLO pré-treinado (10min)
3. `02_yolo_custom.ipynb` - YOLO custom (30-40min) ð
4. `03_cnn_scratch.ipynb` - CNN do zero (20min)
5. `04_comparacao.ipynb` - Análise final (5min)

## 3. Vídeo

Grave 5min mostrando:
- Notebooks executados
- Tabela comparativa
- Sua conclusão

## Resultados

| Modelo | Precisão | Velocidade |
|--------|----------|------------|
| YOLO Trad | ~65% | 205ms |
| YOLO Custom | ~85-90% | 25-30ms |
| **CNN** | **100%** | **1.25ms** ¡ |

CNN é 164x mais rápida!

## Arquivos

- `README.md` - Visão geral
- `RESULTADOS_FINAIS.md` - Resultados
- `experiment_results.log` - Log do treino
- `results/` - Métricas
- `models/` - Modelos salvos
