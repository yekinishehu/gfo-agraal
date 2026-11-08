CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall

all: gfo_solvers

gfo_solvers: gfo_solvers.c case_data.h
	$(CC) $(CFLAGS) -o $@ gfo_solvers.c -lm

run: gfo_solvers
	./gfo_solvers

verify:
	python3 reproduce_experiments.py

clean:
	rm -f gfo_solvers c_periter.dat

.PHONY: all run verify clean
