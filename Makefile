.DEFAULT_GOAL := help
.PHONY: coverage deps lint push test tox help install clean

env:  ## Create and active virtualenv
	python -m venv env

coverage:  ## Run tests with coverage
	coverage erase
	coverage run --include=infostop/* -m pytest -ra
	coverage report -m
	coverage html

deps:  ## Install dependencies
	pip install --upgrade pip
	pip install black coverage pytest build
	if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

test:  ## Run tests
	which pytest
	pytest -ra

pipinstall:  ## Pip install package
	env/bin/pip install ../infostop

install:  ## setup.py install package
	make clean
	python setup.py install

build:  ## Build package to dist/
	make clean
	python -m build --sdist --wheel --outdir dist/

clean:  ## Clean compiled program
	-rm -f *.o
	-rm -f *.so
	-rm -rf *.egg-info*
	-rm -rf ./tmp/
	-rm -rf ./build/
	-rm -rf ./dist/
	-rm -rf ./var/
	-rm -rf ./env/lib/python3.8/site-packages/infostop*
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