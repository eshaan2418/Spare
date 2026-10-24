PY ?= .venv/bin/python
SEED ?= 0
N ?= 50000

.PHONY: help setup test test-fast lint typecheck ci synth verify train-sft train-rl part demo reproduce-quick reproduce-full paper check-numbers scan app

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1

setup:
	uv venv --python 3.12 .venv && uv pip install -e ".[dev,ml]"

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow and not slicer"

lint:
	$(PY) -m ruff check spare tests bench

typecheck:
	$(PY) -m mypy spare

ci: lint typecheck test-fast

synth:
	$(PY) -m spare.synth.build_dataset --n $(N) --seed $(SEED)

verify:
	$(PY) -m bench.verifier_agreement --n 120 --seed $(SEED)

train-sft:
	$(PY) -m spare.cadgen.sft.train

train-rl:
	$(PY) -m spare.cadgen.rl.grpo

part:
	$(PY) -m spare.app.cli part --id $(ID)

demo:
	$(PY) -m spare.app.demo

app:
	$(PY) -m spare.app.server

reproduce-quick:
	bash bench/reproduce_quick.sh

reproduce-full:
	bash bench/reproduce_full.sh

paper:
	$(PY) -m bench.make_figures && $(PY) -m bench.check_numbers

check-numbers:
	$(PY) -m bench.check_numbers

scan:
	$(PY) -m bench.repo_scan
