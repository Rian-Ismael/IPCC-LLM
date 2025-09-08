.PHONY: ingest run eval eval-giskard

INGEST_PDF=data/corpus/IPCC_AR6_SYR_LongerReport.pdf
INDEX_DIR=data/index

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

ingest:
	python ingest/build_index.py --pdf $(INGEST_PDF) --index-dir $(INDEX_DIR)

run:
	streamlit run app/streamlit_app.py

eval:
	python eval/run_ragas.py --index-dir $(INDEX_DIR) --eval-file eval/eval_set.jsonl

eval-giskard:
	python eval/giskard_integration.py