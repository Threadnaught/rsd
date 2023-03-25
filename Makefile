
bin/raw-decode: src/raw-decode.c
	gcc src/raw-decode.c -lavcodec -lavformat -lavutil -g -o bin/raw-decode