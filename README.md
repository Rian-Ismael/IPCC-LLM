# Clima em Foco – IPCC AR6 (SYR)

Assistente de perguntas e respostas baseado no **IPCC AR6 (Relatório Síntese - versão completa)**, construído com **RAG + agentes (LangChain + LangGraph)**.  

O sistema:
- Recupera trechos do relatório oficial.  
- Gera respostas **com citações obrigatórias no formato [p.X]**.  
- Aplica verificações anti-alucinação (*self-check*).  
- Exibe respostas em uma interface Streamlit.  

---

## Funcionalidades

- **RAG (Retrieval-Augmented Generation)** com **Chroma** e embeddings de código aberto (HuggingFace).  
- **Orquestração de agentes com LangGraph**:  
  - **Supervisor** → roteia intenções (único domínio = IPCC).  
  - **Retriever** → busca trechos no índice vetorial.  
  - **Answerer** → gera respostas com citações obrigatórias.  
  - **Self-check** → rejeita respostas sem evidências de suporte.  
  - **Safety** → adiciona avisos automáticos.  
- **Interface Web em Streamlit** (EN).  
- **Citações com links diretos** para o PDF oficial do IPCC.  
- **Avaliação planejada** com **RAGAS**.  

---

## Instalação Local

Requisitos:
- Python 3.11+  
- [Ollama](https://ollama.com/) (para rodar LLMs locais de código aberto).  
- Git e Make.  

Clone o repositório:

```bash
git clone https://github.com/Rian-Ismael/IPCC-LLM.git
cd IPCC-LLM
```

Crie um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

Instale as dependências:

```bash
make install
```

ou diretamente:

```bash
pip install -r requirements.txt
```

---

## Configuração

Crie seu arquivo `.env` baseado no exemplo fornecido:

```bash
cp .env.example .env
```

---

## Indexação

Antes de rodar a interface, indexe o relatório:

```bash
make ingest
```

Isso constrói a base vetorial em `data/index/`.

---

## Executando a Interface (local)

```bash
make run
```

ou diretamente:

```bash
streamlit run app/streamlit_app.py
```

Acesse em [http://localhost:8501](http://localhost:8501).

---

## Executando com Docker + Compose

Construa a imagem e inicie via Docker Compose:

```bash
make build
make up-d
```

Logs:

```bash
make logs
```

Parar:

```bash
make down
```

---

## Avaliação (em andamento)

- **RAGAS** → métricas de *faithfulness* e *answer relevancy*.  
- Um conjunto curado de ~20 perguntas.

---

## Limitações

- Projeto acadêmico de **prova de conceito**.  
- Respostas são **apenas informativas** → não substituem interpretações oficiais do IPCC.  

---

## Licença

Distribuído sob a Licença MIT — veja [LICENSE](LICENSE).  

---

## Citação

Se você utilizar este projeto em trabalhos acadêmicos:

```bibtex
@software{clima_em_foco_2025,
  authors = {Elias de Melo, Rian Ismael; Veríssimo, Victor de Sousa},
  title = {Clima em Foco – IPCC AR6 (SYR) Assistant with RAG + Agents},
  year = {2025},
  url = {https://github.com/Rian-Ismael/IPCC-LLM}
}
```
