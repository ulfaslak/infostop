SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage deps test install build clean help

env:  ## Create a virtualenv in ./env
	python3 -m venv env

coverage:  ## Run tests with coverage
	coverage erase
	coverage run --include=infostop/* -m pytest -ra
	coverage report -m
	coverage html

deps:  ## Install dependencies
	pip install --upgrade pip
	pip install black coverage pytest build
	if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

test:  ## Run tests against the installed package, like CI (run make install first)
	@tmp=$$(mktemp -d) && cp -R tests "$$tmp/" && cd "$$tmp" && \
		python -c "import infostop; print('Testing', infostop.__file__)" && \
		python -m pytest -ra tests; \
		status=$$?; rm -rf "$$tmp"; exit $$status

install:  ## Build and install the package, replacing any earlier install
	$(MAKE) clean
	pip install .

build:  ## Build package to dist/
	$(MAKE) clean
	python -m build --sdist --wheel --outdir dist/

clean:  ## Clean compiled program
	-rm -f *.o
	-rm -f *.so
	-rm -rf *.egg-info*
	-rm -rf ./tmp/
	-rm -rf ./build/
	-rm -rf ./dist/
	-rm -rf ./var/
	-rm -rf ./infostop/__pycache__/
	-rm -rf ./tests/__pycache__/

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done