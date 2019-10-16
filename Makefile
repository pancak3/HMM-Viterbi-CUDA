CC = gcc
CFLAGS = -std=c99

BINDIR = bin/
SRCDIR = src/
UTILDIR = util/

.PHONY: all util
all: driver sequential util
util: generator

$(BINDIR):
	mkdir -p $(BINDIR)

driver: $(BINDIR) $(SRCDIR)driver.c sequential
	$(CC) $(CFLAGS) -o $(BINDIR)driver $(SRCDIR)driver.c $(SRCDIR)viterbi_sequential.c

sequential: $(BINDIR) $(SRCDIR)viterbi_sequential.c
	$(CC) $(CFLAGS) -o $(BINDIR)viterbi_sequential $(SRCDIR)viterbi_sequential.c

generator: $(BINDIR) $(UTILDIR)generator.c
	$(CC) $(CFLAGS) -o $(BINDIR)generator $(UTILDIR)generator.c

