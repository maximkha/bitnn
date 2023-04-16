HIGH_OPT="normal"

run: clean test
	./test > out

bench: clean test run
	bench './test'

clean:
	rm test.cpp || true
	rm test || true
	rm -rfd test.dSYM || true
	rm out || true

test: test.cpp bitnn.cpp
ifeq ($(HIGH_OPT), "fast")
	g++ test.cpp -Ofast -march=native -flto -funroll-loops -mtune=native -o test
else ifeq ($(HIGH_OPT), "debug")
	g++ test.cpp -Ofast -march=native -flto -funroll-loops -mtune=native -g -o test
else
	g++ test.cpp -O3 -o test
endif

test.cpp: gen.py
	python3 gen.py > test.cpp
