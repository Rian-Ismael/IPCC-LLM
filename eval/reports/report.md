# Resultado de Avaliação (RAGAS)

- **Amostras**: 20
- **Faithfulness (média)**: 0.97
- **Answer Relevancy (média)**: 0.8254
- **Context Precision (média)**: 0.5893
- **Context Recall (média)**: 0.8643

## Latência (ms)
- média: 56657 | min: 42034 | max: 99014
- p50: 52518 | p95: 86664

## Footprint (média aproximada do processo)
- **Pico de memória**: 3125.3 MB
- **CPU média do processo**: 63.7%

## Auditoria do Retriever (gold_page)
- **gold_hit_rate**: 0.00%

## Observações
- As métricas de contexto (precision/recall) vêm do RAGAS puro.
- O footprint é amostrado durante toda a execução (0.5s), aproximado para seu processo.
- A auditoria gold_hit_rate não altera as métricas do RAGAS; serve apenas para checar o retriever.
