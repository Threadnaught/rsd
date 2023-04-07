
build-here: src/* pysrc/setup.py
	python3 pysrc/setup.py build_ext --inplace && cp arsd/* pysrc/

install:
	if ! python3 pysrc/setup.py install --user; then \
		echo ""; \
		echo "------------------------------------------------------------------------------------------------------"; \
		echo ""; \
		echo "\033[1mIt would appear that ARSD failed to install.\033[0m "; \
		echo "Most likely, you have not installed the deps."; \
		echo "See https://github.com/Threadnaught/arsd#install-dependencies"; \
		echo ""; \
	fi