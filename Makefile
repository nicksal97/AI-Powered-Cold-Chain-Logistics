.PHONY: install up down seed api ui test lint fmt eval-retrieval eval-agent

install:        ## local dev install
	pip install -e ".[dev,ui]"

up:             ## full stack: db, seed, api, ui
	docker compose up --build -d

down:
	docker compose down

seed:           ## migrate + load telemetry + index SOPs + detect incidents
	coldchain seed

api:
	uvicorn coldchain.api.app:app --reload --port 8000

ui:
	streamlit run ui/app.py

test:
	pytest --cov=coldchain

lint:
	ruff check . && ruff format --check .

fmt:
	ruff check --fix . && ruff format .

eval-retrieval:
	python evals/run_evals.py retrieval

eval-agent:     ## needs COLDCHAIN_LLM_API_KEY and a seeded db
	python evals/run_evals.py agent
