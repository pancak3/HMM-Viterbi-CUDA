CC = gcc
CFLAGS = -std=c99

BINDIR = bin/
SRCDIR = src/
UTILDIR = util/

.PHONY: all util
all: sequential util
util: generator

$(BINDIR):
	mkdir -p $(BINDIR)

sequential: $(BINDIR) $(SRCDIR)viterbi_sequential.c
	$(CC) $(CFLAGS) -o $(BINDIR)viterbi_sequential $(SRCDIR)viterbi_sequential.c

generator: $(BINDIR) $(UTILDIR)generator.c
	$(CC) $(CFLAGS) -o $(BINDIR)generator $(UTILDIR)generator.c

