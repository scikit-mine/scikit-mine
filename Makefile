coverage:
	pytest --cov-report term-missing --cov=skmine --cov-config=.coveragerc skmine

setup:
	python setup.py install

clean: clean_doc
	$(RM) *.cpp
	$(RM) *.so
	$(RM) **/*.cpp
	$(RM) **/*.so
	find . -name "*.pyc" -exec rm -f {} \;


clean_doc:
	$(RM) -f docs/skmine.rst docs/modules.rst


docs: setup clean_doc
	sphinx-apidoc -o docs/ skmine/ tests/*
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

docs-github: docs
	cd docs/_build/html
	tar czf /tmp/html.tgz .
	cd ../../../
	git checkout gh-pages
	git rm -rf .
	tar xzf /tmp/html.tgz
	git add .
	git commit -m 'update doc'
	git push official gh-pages

sort_python:
	isort -sl -rc -y   # single line imports for cleaner versionning via git
