coverage:
	pytest --cov-report term-missing --cov=skmine --cov-config=.coveragerc skmine

clean:
	$(RM) *.cpp
	$(RM) *.so
	$(RM) **/*.cpp
	$(RM) **/*.so
	find . -name "*.pyc" -exec rm -f {} \;



clean_py_files:
	isort -sl -rc -y   # single line imports for cleaner versionning via git
