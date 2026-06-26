# PyStorm task runner (Linux / macOS / Git Bash).
# Windows users: use tasks.ps1 (same verbs).
#
#   make install            install everything (deps + modules + kernels)
#   make doctor             check the environment
#   make build              build all C++ kernels
#   make test               run every module's test suite
#   make run M=rss          run a module launcher (M = ahd|sca|lcs|jdm|pot|pst|rss|csh)
#   make run M=rss ARGS="--mode optimal --scope regional"
#
PYTHON ?= python

# acronym -> module directory
ahd := augmented_hurricane_database
sca := storm_climatology_analysis
lcs := life_cycle_simulation
jdm := joint_distribution_model
pot := peaks_over_threshold
pst := probabilistic_simulation_technique
rss := reduced_storm_suite
csh := coastal_storm_hydrograph
MODULES := $(ahd) $(sca) $(lcs) $(jdm) $(pot) $(pst) $(rss) $(csh)

BUILDS := \
  modules/$(pot)/backend/engines/cpp/build.py \
  modules/$(pst)/backend/engines/build.py \
  modules/$(rss)/backend/engines/cpp/build.py \
  modules/$(jdm)/backend/engines/cpp/build.py \
  modules/$(ahd)/backend/engines/cpp/build.py

.PHONY: help install doctor build test run

help:
	@echo "targets: install | doctor | build | test | run M=<acro> [ARGS=...]"
	@echo "acronyms: ahd sca lcs jdm pot pst rss csh"

install:
	@./install.sh

doctor:
	@$(PYTHON) check_env.py

build:
	@for b in $(BUILDS); do echo "build $$b"; $(PYTHON) $$b || echo "WARN: $$b failed (fallback)"; done

test:
	@for m in $(MODULES); do echo "== test $$m =="; ( cd modules/$$m && $(PYTHON) -m pytest -q ) || exit $$?; done

run:
	@d="$($(M))"; \
	if [ -z "$$d" ]; then echo "usage: make run M=<ahd|sca|lcs|jdm|pot|pst|rss|csh> [ARGS=...]"; exit 2; fi; \
	$(PYTHON) modules/$$d/run_$$d.py $(ARGS)
