coverage:
	pytest --cov-report term-missing --cov=skmine --cov-config=.coveragerc skmine

clean: clean_doc
	$(RM) *.cpp
	$(RM) *.so
	$(RM) **/*.cpp
	$(RM) **/*.so
	find . -name "*.pyc" -exec rm -f {} \;


clean_doc:
	$(RM) -f docs/skmine.rst docs/modules.rst


docs: clean_doc
	sphinx-apidoc -o docs/ skmine/ tests/*
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

sort_python:
	isort -sl -rc -y   # single line imports for cleaner versionning via git
