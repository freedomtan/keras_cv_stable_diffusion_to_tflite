CFLAGS=-O2 -D__TEST_BPE__ -std=c++17
TARGET_BINARY=test_bpe test_random test_sched

all: ${TARGET_BINARY}

test_bpe: bpe.cc
	$(CXX) ${CFLAGS} $< -o $@

test_random: test_random.cc
	$(CXX) ${CFLAGS} $< -o $@

test_sched: scheduling_util.cc
	$(CXX) ${CFLAGS} $< -o $@

clean:
	rm ${TARGET_BINARY}
