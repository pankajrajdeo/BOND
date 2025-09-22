PYTHON ?= python3
PIP ?= pip

.PHONY: install dev lint test serve query build-faiss gen-sqlite ingest-umls ingest-lungmap

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

serve:
	BOND_ALLOW_ANON=1 bond-serve

query:
	bond-query --query "T-cell" --field_name "cell type" --verbose

build-faiss:
	bond-build-faiss --sqlite_path assets/ontologies.sqlite --assets_path assets

gen-sqlite:
	bond-generate-sqlite

ingest-umls:
	bond-ingest-umls

ingest-lungmap:
	bond-ingest-lungmap

lint:
	ruff check . || true

test:
	pytest -q || true

