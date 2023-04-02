
build: src/* pysrc/setup.py
	python3 pysrc/setup.py build_ext --inplace && cp arsd/* pysrc/
