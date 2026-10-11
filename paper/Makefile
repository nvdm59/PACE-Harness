# Build the technical report with tectonic (fetches TeX packages on first use).
#   make            -> main.pdf
#   make figures    -> regenerate paper/figures/* from the committed result files
#   make clean
TECTONIC ?= /opt/homebrew/bin/tectonic
PY       ?= ../.venv-crmarena/bin/python

all: main.pdf

main.pdf: main.tex refs.bib figures/tab_continual_days.tex
	$(TECTONIC) --keep-logs --keep-intermediates main.tex

figures figures/tab_continual_days.tex:
	$(PY) make_figures.py

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc

.PHONY: all figures clean
