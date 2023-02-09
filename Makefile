CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

LCS_TEST_EXE_NAME = lcs_test
LCS_TEST_SOURCES = src/main.cpp src/algorithm_registry.cpp

build:
	$(CXX) $(CXXFLAGS) -o $(LCS_TEST_EXE_NAME) $(LCS_TEST_SOURCES)

clean:
	rm -f $(LCS_TEST_EXE_NAME)












