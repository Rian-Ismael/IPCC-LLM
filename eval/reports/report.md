# Resultado de Avaliação (RAGAS)

- **Amostras**: 20
- **Faithfulness (média)**: 0.9582
- **Answer Relevancy (média)**: 0.8231
- **Context Precision (média)**: 0.7247
- **Context Recall (média)**: 0.4262

## Latência (ms)
- média: 14690 | min: 10273 | max: 18243
- p50: 14901 | p95: 18061

## Footprint (média aproximada do processo)
- **Pico de memória**: 794.1 MB
- **CPU média do processo**: 10.7%

## Auditoria do Retriever (gold_page)
- **gold_hit_rate**: 0.00%

## Observações
- As métricas de contexto (precision/recall) vêm do RAGAS puro.
- O footprint é amostrado durante toda a execução (0.5s), aproximado para seu processo.
- A auditoria gold_hit_rate não altera as métricas do RAGAS; serve apenas para checar o retriever.
