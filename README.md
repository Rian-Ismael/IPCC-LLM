# Clima em Foco — IPCC AR6 (SYR Longer Report)

**PoC** de RAG + Agentes (LangGraph) para responder perguntas sobre o IPCC AR6 (Relatório de Síntese — Longer Report) com **citações obrigatórias**, **self-check** e **avaliação automática via Giskard RAGET**.

## Stack
Python 3.11, LangChain/LangGraph, Chroma, Sentence-Transformers, Streamlit, Giskard, Ollama (open-weights LLM).

## Como rodar
1. Coloque o PDF em `data/corpus/IPCC_AR6_SYR_LongerReport.pdf`.
2. Ingestão do corpus:
   ```bash
   make ingest
   ```


App (UI):

make run

Avaliação automática (Giskard):

make eval-giskard

Saída: eval/reports/ipcc_raget_report.html.

Arquitetura (LangGraph)

Supervisor → Retriever → Answerer (Ollama) → Self-check → Safety.

Citações: páginas no formato [p.X] + trechos listados.

Self-check: recusa se não houver citação ou evidência suficiente.

Safety: adiciona disclaimer informativo.

Avaliação (rubrica)

Conjunto de 25 perguntas em eval/ipcc_testset.jsonl (gerado automaticamente se ausente).

Métricas (via Giskard/RAGAS): Context Precision, Context Recall.

Latência: registrada por pergunta no texto do answer (suficiente para sumarizar no relatório).

Ética & Segurança

Conteúdo informativo baseado no IPCC AR6. Não substitui leitura/interpretação oficial.

Reprodutibilidade

requirements.txt, Dockerfile, Makefile, passos de ingestão.

LLM local via Ollama configurável por .env (variável OLLAMA_MODEL).


ifeq ($(OS),Windows_NT)
VENV_PY := .venv\Scripts\python.exe
else
VENV_PY := .venv/bin/python
endif

PYTHON := $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)

.PHONY: setup ingest run eval eval-giskard

setup:
	$(PYTHON) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) ingest/build_index.py --pdf data/corpus/IPCC_AR6_SYR_LongerReport.pdf --index-dir data/index

run:
	$(PYTHON) app/main.py

eval:
	$(PYTHON) eval/run_ragas.py

eval-giskard:
	$(PYTHON) eval/run_giskard.py
