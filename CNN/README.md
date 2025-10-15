# FIAP Fase 6 - Comparação YOLO vs CNN

> Classificação de imagens: Coffee ☕ vs Mountains 🏔️

## O que fizemos

Comparamos 3 abordagens de visão computacional:
1. YOLO pré-treinado (zero setup)
2. YOLO customizado (treino com nosso dataset)
3. CNN do zero (arquitetura própria)

## Resultados

| Modelo | Precisão | Tempo Treino | Inferência |
|--------|----------|--------------|------------|
| YOLO Tradicional | ~65% | 0s | 205ms |
| YOLO Custom | ~85-90% | 15-40min | 25-30ms |
| **CNN do Zero** | **100%** ⭐ | 3min | **1.25ms** ⚡ |

**CNN é 164x mais rápida que YOLO!**

## Como executar

```bash
# Setup
source .venv/bin/activate

# Teste rápido (3 minutos)
python run_experiments.py

# Notebooks completos
jupyter notebook
# Execute: 01, 02, 03, 04 em ordem
```

## Estrutura

```
├── src/              # Código Python
├── notebooks/        # 4 notebooks Jupyter
├── results/          # Resultados salvos
└── models/           # Modelos treinados
```

## Para o trabalho

1. Execute os notebooks
2. Grave vídeo de 5min
3. Suba no GitHub
4. Envie no portal FIAP

## Conclusão

**Melhor escolha**: Depende do caso
- Prototipagem rápida? → YOLO Tradicional
- Melhor precisão? → YOLO Custom
- Velocidade máxima? → CNN do Zero (164x mais rápida!)

---

**Autor**: Luiz | **FIAP 2025**
