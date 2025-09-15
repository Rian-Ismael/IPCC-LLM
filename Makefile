.PHONY: help build up up-d down logs ps sh ingest eval eval-giskard restart

COMPOSE ?= docker compose
APP_SERVICE ?= app
INGEST_SERVICE ?= ingest

help:
	@echo "Usage:"
	@echo "  make build         # Build images"
	@echo "  make ingest        # (Re)build the vector index inside a one-off container"
	@echo "  make up            # Start app in foreground (build if needed)"
	@echo "  make up-d          # Start app in background (detached)"
	@echo "  make logs          # Tail logs from the app service"
	@echo "  make ps            # Show service status"
	@echo "  make sh            # Shell into the running app container"
	@echo "  make eval          # Run RAGAS eval inside container"
	@echo "  make eval-giskard  # Run Giskard eval inside container"
	@echo "  make down          # Stop and remove containers"
	@echo "  make restart       # Restart the app service"

build:
	$(COMPOSE) build

ingest:
	$(COMPOSE) run --rm $(INGEST_SERVICE)

up:
	$(COMPOSE) up --build

up-d:
	$(COMPOSE) up -d --build

logs:
	$(COMPOSE) logs -f $(APP_SERVICE)

ps:
	$(COMPOSE) ps

sh:
	$(COMPOSE) exec $(APP_SERVICE) /bin/sh

eval:
	# Uses INDEX_DIR from .env if present; defaults to data/index
	$(COMPOSE) run --rm $(APP_SERVICE) python -m eval.run_ragas --index-dir $${INDEX_DIR:-data/index} --eval-file eval/eval_set.jsonl

eval-giskard:
	$(COMPOSE) run --rm $(APP_SERVICE) python -m eval.giskard_integration

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart $(APP_SERVICE)

run:
	python -m streamlit run app/streamlit_app.py