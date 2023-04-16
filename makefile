run: test
	./test > out

clean:
	rm test.cpp
	rm test
	rm out

test: test.cpp bitnn.cpp
	g++ test.cpp -O3 -o test

test.cpp: gen.py
	python3 gen.py > test.cpp
