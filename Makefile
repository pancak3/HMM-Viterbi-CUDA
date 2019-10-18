CC = gcc
CFLAGS = -std=c99 -DDEBUG
# CFLAGS = -std=c99
LINKER = $(CC)
LFLAGS = -lm

BINDIR = bin/
OBJDIR = obj/
SRCDIR = src/
UTILDIR = util/

.PHONY: all util
all: $(BINDIR)driver util
util: $(BINDIR)generator

$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR)driver: $(BINDIR) $(OBJDIR)driver.o $(OBJDIR)viterbi_sequential.o 
	$(LINKER) -o $(BINDIR)driver $(OBJDIR)driver.o $(OBJDIR)viterbi_sequential.o $(LFLAGS)

$(OBJDIR)driver.o: $(OBJDIR) $(SRCDIR)driver.c
	$(CC) $(CFLAGS) -o $(OBJDIR)driver.o -c $(SRCDIR)driver.c

$(OBJDIR)viterbi_sequential.o: $(OBJDIR) $(SRCDIR)viterbi_sequential.c
	$(CC) $(CFLAGS) -o $(OBJDIR)viterbi_sequential.o -c $(SRCDIR)viterbi_sequential.c

$(BINDIR)generator: $(BINDIR) $(UTILDIR)generator.c
	$(CC) $(CFLAGS) -o $(BINDIR)generator $(UTILDIR)generator.c

.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(BINDIR)

