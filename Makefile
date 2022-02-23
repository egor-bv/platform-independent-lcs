CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17 -xCORE-AVX2

LCS_TEST_EXE_NAME = lcs_test
LCS_TEST_SOURCES = src/main.cpp


build:
	$(CXX) $(CXXFLAGS) -o $(LCS_TEST_EXE_NAME) $(LCS_TEST_SOURCES)

run_cpu:
	./$(LCS_TEST_EXE_NAME)

run_gpu:
	./$(LCS_TEST_EXE_NAME) gpu

clean:
	rm -f $(LCS_TEST_EXE_NAME)












