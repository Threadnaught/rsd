
build-here: src/* pysrc/setup.py
	python3 pysrc/setup.py build_ext --inplace && mv ./rsd.cpython* pysrc

install:
	rm -f pysrc/rsd.cpython*
	if ! python3 pysrc/setup.py install; then \
		echo ""; \
		echo "------------------------------------------------------------------------------------------------------"; \
		echo ""; \
		echo "\033[1mIt would appear that RSD failed to install.\033[0m "; \
		echo "Most likely, you have not installed the deps."; \
		echo "See https://github.com/Threadnaught/rsd#install-dependencies"; \
		echo ""; \
	fi
