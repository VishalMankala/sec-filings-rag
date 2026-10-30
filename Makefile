.PHONY: install test lint ingest eval eval-retrieval serve ui

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[models,ui,dev]"

lint:
	ruff check src tests app.py

test:
	pytest

ingest:             ## download the 10-Ks from EDGAR and build the index
	secrag ingest

eval-retrieval:     ## no API key needed
	secrag eval --retrieval-only

eval:               ## needs GEMINI_API_KEY
	secrag eval

serve:
	secrag serve

ui:
	streamlit run app.py
