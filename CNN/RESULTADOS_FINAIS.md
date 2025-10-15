# Resultados - FIAP Fase 6

## Experimentos Realizados

Comparamos 3 abordagens para classificar Coffee vs Mountains:

### 1. YOLO Tradicional
- Modelo pré-treinado (zero setup)
- **Inferência**: 205ms
- **Precisão**: ~65% (genérica, 80 classes COCO)
- **Facilidade**: ⭐⭐⭐⭐⭐

### 2. CNN do Zero
- Arquitetura própria, treino do zero
- **Treino**: 3 minutos (10 épocas)
- **Inferência**: **1.25ms** ⚡
- **Precisão**: **100%** no teste ⭐
- **Facilidade**: ⭐⭐⭐

### 3. YOLO Custom
- Fine-tuning com nosso dataset
- **Treino**: 15-40min (30-60 épocas)
- **Inferência**: 25-30ms
- **Precisão estimada**: 85-90%
- **Facilidade**: ⭐⭐⭐⭐

## Conclusão

**CNN do Zero venceu:**
- ✅ 164x mais rápida que YOLO (1.25ms vs 205ms)
- ✅ 100% de acurácia no teste
- ✅ Treino rápido (3 minutos)

**Quando usar cada um:**
- **YOLO Tradicional**: Prototipagem rápida, sem treino
- **YOLO Custom**: Múltiplos objetos, detecção + classificação
- **CNN do Zero**: Classificação simples, velocidade crítica

## Para o Trabalho

**Resposta para "Comparação de Abordagens"**:

1. **Facilidade**: YOLO Trad (zero setup) > YOLO Custom > CNN
2. **Precisão**: CNN (100%) > YOLO Custom (85-90%) > YOLO Trad (65%)
3. **Velocidade**: CNN (1.25ms) >> YOLO Custom (25ms) >> YOLO Trad (205ms)
4. **Recomendação**: Para Coffee vs Mountains, **CNN é a melhor escolha**

## Arquivos

- `run_experiments.py` - Script com os testes
- `experiment_results.log` - Log completo do treino
- `results/` - Métricas salvas
- `models/cnn_scratch_quick.pth` - Modelo CNN treinado

---

**Luiz - Parte 4: Abordagens Alternativas**
