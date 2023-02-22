CFLAGS=-O2 -D__TEST_BPE__ -std=c++17
all: test_bpe

test_bpe: bpe.cc
	$(CXX) ${CFLAGS} $< -o $@

clean:
	rm test_bpe
