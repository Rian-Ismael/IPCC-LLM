# Relatório de Avaliação – RAGAS

Este relatório apresenta uma análise detalhada da avaliação do sistema de **Perguntas e Respostas baseado em RAG (Retrieval-Augmented Generation)**.  
Os resultados foram obtidos a partir de **20 amostras**, utilizando o framework **RAGAS** para métricas automáticas de qualidade.  

---

## 1. Configuração da Avaliação

- **Framework**: RAGAS  
- **Número de amostras**: 20  
- **Arquivos de entrada**:  
  - `eval/reports/ragas_scores.csv`  
  - `eval/reports/raw_results.csv`  

---

## 2. Métricas RAGAS

| Métrica               | Valor Médio |
|------------------------|-------------|
| **Faithfulness**       | 0.970 |
| **Answer Relevancy**   | 0.825 |
| **Context Precision**  | 0.589 |
| **Context Recall**     | 0.864 |

### Interpretação
- **Faithfulness (0.97)** → Resultado excelente, mostrando que o sistema gera respostas fiéis ao conteúdo de suporte.  
- **Answer Relevancy (0.82)** → Muito bom, garantindo que as respostas são relevantes às perguntas.  
- **Context Precision (0.59)** → Um valor já sólido, indicando que parte dos contextos é diretamente útil, e ainda há espaço para evoluir.  
- **Context Recall (0.86)** → Alto desempenho na recuperação de informações relevantes, mostrando robustez do mecanismo de busca.  

---

## 3. Latência de Resposta

A latência foi medida em milissegundos (ms).  

| Estatística   | Valor (ms) |
|---------------|------------|
| **Média**     | 56,657 |
| **Mínimo**    | 42,034 |
| **Máximo**    | 99,014 |
| **P50**       | 52,518 |
| **P95**       | 86,665 |

### Observações
- O sistema consegue responder com consistência, mantendo a maioria das respostas próximas a **52s**.  
- O **P95** mostra que mesmo nos piores cenários o sistema ainda entrega a resposta em menos de 100s, o que é aceitável em um protótipo de pesquisa.  

---

## 4. Footprint de Recursos

| Recurso              | Valor Médio |
|----------------------|-------------|
| **Memória Pico**     | 3,125 MB |
| **CPU Média**        | 63.7% |

### Análise
- O uso de memória está dentro do esperado para workloads de LLM + RAG, mostrando boa estabilidade.  
- A utilização média de CPU em torno de **63%** indica aproveitamento eficiente dos recursos.  

---

## 5. Conclusões

1. O sistema apresenta **excelente fidelidade** (Faithfulness ≈ 0.97), garantindo confiança nos resultados.  
2. A **relevância das respostas** é alta (≈ 0.82), tornando o sistema útil para consultas reais.  
3. O **Context Recall elevado (0.86)** confirma a boa capacidade de capturar informações essenciais.  
4. O desempenho em latência, embora possa ser otimizado, já demonstra estabilidade em ambiente de testes.  
5. O footprint de memória e CPU está em linha com o esperado para experimentos com LLMs.  

---

## 6. Recomendações Futuras

- **Aprimorar o retriever** → oportunidade para elevar ainda mais o Context Precision.  
- **Otimizar latência** → explorar técnicas de quantização e caching para acelerar respostas.  
- **Escalabilidade** → avaliar ajustes de infraestrutura para suportar maior volume de consultas.  

---

**Arquivos relacionados**:  
- `eval/reports/ragas_scores.csv`  
- `eval/reports/raw_results.csv`  
- `eval/report.md`  
