# Clima em Foco – IPCC AR6 (SYR)

Assistente de perguntas e respostas baseado no **IPCC AR6 (SYR Longer Report)**, utilizando **RAG + agentes (LangChain + LangGraph)**.  
O sistema recupera trechos do relatório oficial, gera respostas **com citações obrigatórias [p.X]** e aplica verificações anti-alucinação.

---

## 🚀 Funcionalidades
- **RAG (Retrieval-Augmented Generation)** com Chroma + embeddings OSS.
- **Orquestração com agentes LangGraph**:
  - Supervisor → roteia intenções.
  - Retriever → busca trechos no índice vetorial.
  - Answerer → gera respostas com citações obrigatórias.
  - Self-check → recusa respostas sem evidência.
  - Safety → adiciona disclaimers.
- **UI em Streamlit** para perguntas (PT/EN).
- **Citações com links diretos para o PDF oficial do IPCC.**
- **Avaliação planejada com RAGAS e Giskard**.

---

## ⚙️ Instalação

Pré-requisitos:
- Python 3.10+
- [Ollama](https://ollama.com/) instalado (para rodar modelos LLM open-weights).
- Git + virtualenv.

Clone o projeto:
```bash
git clone https://github.com/seuusuario/ipcc-llm.git
cd ipcc-llm
```

Crie ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

Instale dependências:
```bash
pip install -r requirements.txt
```

---

## Configuração

Copie o arquivo `.env.example` para `.env` e ajuste conforme necessário:

```bash
cp .env.example .env
```

Exemplo de variáveis:
```ini
OLLAMA_MODEL=llama3.2:3b-instruct-q4_K_M
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
INDEX_DIR=data/index
PDF_PATH=data/corpus/IPCC_AR6_SYR_LongerReport.pdf
```

---

## Execução

Para construir o índice:
```bash
make ingest
```

Para rodar a UI Streamlit:
```bash
streamlit run app/streamlit_app.py
```

---

## Avaliação (em progresso)
- **RAGAS**: métricas de *faithfulness* e *answer relevancy*.
- **Giskard**: testes de robustez e relevância das respostas.

---

## Limitações
- Projeto acadêmico de prova de conceito.
- Conteúdo apenas **informativo**.  
- Não substitui interpretações oficiais do IPCC.

---

## Licença
Distribuído sob a licença MIT — veja [LICENSE](LICENSE).

---

## Citação
Se utilizar este projeto em trabalhos acadêmicos:

```bibtex
@software{clima_em_foco_2025,
  author = {Elias de Melo, Rian Ismael},
  title = {Clima em Foco – IPCC AR6 (SYR) Assistant with RAG + Agents},
  year = {2025},
  url = {https://github.com/Rian-Ismael/IPCC-LLM}
}
```
