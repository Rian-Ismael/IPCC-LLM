# Climate in Focus – IPCC AR6 (SYR)

Question-answering assistant based on the **IPCC AR6 (SYR Longer Report)**, built with **RAG + agents (LangChain + LangGraph)**.  

The system:
- Retrieves passages from the official report.  
- Generates answers **with mandatory citations in the format [p.X]**.  
- Applies anti-hallucination checks (self-check).  
- Displays answers in a Streamlit interface.  

---

## Features

- **RAG (Retrieval-Augmented Generation)** with **Chroma** and open-source embeddings (HuggingFace).
- **Agent orchestration with LangGraph**:
  - **Supervisor** → routes intents (single domain = IPCC).
  - **Retriever** → fetches passages from the vector index.
  - **Answerer** → generates answers with mandatory citations.
  - **Self-check** → rejects answers without supporting evidence.
  - **Safety** → adds automatic disclaimers.
- **Web UI in Streamlit** (EN).
- **Citations with direct links** to the official IPCC PDF.
- **Planned evaluation** with **RAGAS**.

---

## Local Installation

Requirements:
- Python 3.11+
- [Ollama](https://ollama.com/) (for running local open-weight LLMs).  
- Git and Make.

Clone the repository:

```bash
git clone https://github.com/Rian-Ismael/IPCC-LLM.git
cd IPCC-LLM
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create your `.env` file based on the provided example:

```bash
cp .env.example .env
```

---

## Indexing

Before running the interface, index the report:

```bash
make ingest
```

This builds the vector database in `data/index/`.

---

## Running the UI (local)

```bash
make run
```

or directly:

```bash
streamlit run app/streamlit_app.py
```

Access at [http://localhost:8501](http://localhost:8501).

---

## Running with Docker + Compose

Build the image and start via Docker Compose:

```bash
make build
make up-d
```

Logs:

```bash
make logs
```

Stop:

```bash
make down
```

---

## Evaluation (in progress)

- **RAGAS** → metrics for *faithfulness* and *answer relevancy*.  
- A curated set of ~20 manually annotated questions is in progress.

---

## Limitations

- Academic **proof of concept** project.  
- Answers are **informational only** → do not replace official IPCC interpretations.  
- Small LLMs may occasionally return **empty outputs**.

---

## License

Distributed under the MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use this project in academic work:

```bibtex
@software{climate_in_focus_2025,
  author = {Elias de Melo, Rian Ismael},
  title = {Climate in Focus – IPCC AR6 (SYR) Assistant with RAG + Agents},
  year = {2025},
  url = {https://github.com/Rian-Ismael/IPCC-LLM}
}
```
