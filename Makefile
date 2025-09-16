# Use:
#   Local (sem Docker):
#     make venv           # cria .venv
#     make install        # instala deps no .venv
#     make ingest         # gera índice vetorial
#     make run            # inicia Streamlit local
#     make eval           # roda RAGAS local
#     make eval-giskard   # roda integração Giskard local
#
#   Docker / Compose:
#     make build          # build das imagens
#     make ingest-docker  # ingestão dentro de um container one-off
#     make up-d           # sobe app em segundo plano
#     make logs           # logs do app
#     make down           # derruba os services
#     make eval-docker    # RAGAS dentro do container
#     make sh             # shell dentro do container do app

.PHONY: help venv install ingest run eval eval-giskard \
        build up up-d down logs ps sh ingest-docker eval-docker eval-giskard-docker \
        restart clean-index clean-venv clean-docker

# Variáveis gerais 
PY ?= python
PDF_PATH ?= data/corpus/IPCC_AR6_SYR_LongerReport.pdf
INDEX_DIR ?= data/index

# Variáveis Docker/Compose 
COMPOSE ?= docker compose
APP_SERVICE ?= app
INGEST_SERVICE ?= ingest

help:
	@echo ""
	@echo "Local (sem Docker):"
	@echo "  make venv            - cria ambiente virtual .venv"
	@echo "  make install         - instala dependências no .venv"
	@echo "  make ingest          - gera índice (usa PDF_PATH e INDEX_DIR)"
	@echo "  make run             - inicia Streamlit local (http://localhost:8501)"
	@echo "  make eval            - executa RAGAS local"
	@echo "  make eval-giskard    - executa integração Giskard local"
	@echo ""
	@echo "Docker / Compose:"
	@echo "  make build           - build das imagens do docker-compose"
	@echo "  make ingest-docker   - ingestão (one-off container) usando docker-compose"
	@echo "  make up              - sobe app em primeiro plano (rebuild se preciso)"
	@echo "  make up-d            - sobe app em segundo plano (detached)"
	@echo "  make logs            - tail dos logs do serviço '$(APP_SERVICE)'"
	@echo "  make ps              - status dos serviços"
	@echo "  make sh              - shell dentro do container '$(APP_SERVICE)'"
	@echo "  make eval-docker     - RAGAS dentro do container"
	@echo "  make eval-giskard-docker - Giskard dentro do container"
	@echo "  make down            - derruba os serviços"
	@echo "  make restart         - reinicia o serviço '$(APP_SERVICE)'"
	@echo ""
	@echo "Limpeza:"
	@echo "  make clean-index     - apaga $(INDEX_DIR)"
	@echo "  make clean-venv      - apaga .venv"
	@echo "  make clean-docker    - remove imagens/volumes parados (cuidado)"
	@echo ""

# Local 
venv:
	$(PY) -m venv .venv

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

ingest:
	$(PY) -m ingest.build_index --pdf "$(PDF_PATH)" --index-dir "$(INDEX_DIR)"

run:
	$(PY) -m streamlit run app/streamlit_app.py

eval:
	$(PY) -m eval.run_ragas

eval-giskard:
	$(PY) -m eval.giskard_integration

# Docker/Compose 
build:
	$(COMPOSE) build

ingest-docker:
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

eval-docker:
	$(COMPOSE) run --rm $(APP_SERVICE) $(PY) -m eval.run_ragas

eval-giskard-docker:
	$(COMPOSE) run --rm $(APP_SERVICE) $(PY) -m eval.giskard_integration

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart $(APP_SERVICE)

# Limpeza
clean-index:
	-$(PY) -c "import shutil, os; shutil.rmtree(os.environ.get('INDEX_DIR', '$(INDEX_DIR)'), ignore_errors=True)"

clean-venv:
	-$(PY) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"

clean-docker:
	-docker system prune -f
