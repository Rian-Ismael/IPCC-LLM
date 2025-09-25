# Resultado de Avaliação (RAGAS)

- **Amostras**: 1
- **Faithfulness (média)**: 1.0
- **Answer Relevancy (média)**: 0.9072
- **Context Precision (média)**: 0.6821
- **Context Recall (média)**: 1.0

## Latência (ms)
- média: 32776 | min: 32776 | max: 32776
- p50: 32776 | p95: 32776

## Footprint (média aproximada do processo)
- **Pico de memória**: 1069.1 MB
- **CPU média do processo**: 15.5%

## Auditoria do Retriever (gold_page)
- **gold_hit_rate**: 0.00%

## Observações
- As métricas de contexto (precision/recall) vêm do RAGAS puro.
- O footprint é amostrado durante toda a execução (0.5s), aproximado para seu processo.
- A auditoria gold_hit_rate não altera as métricas do RAGAS; serve apenas para checar o retriever.
