#pragma once
// general utility functions with minimal dependecies


// reimplementing some standard functions so they can be compiled inside SYCL kernel code

template<typename T>
inline T Max(T x, T y)
{
	return x > y ? x : y;
}

template<typename T>
inline T Min(T x, T y)
{
	return x < y ? x : y;
}

template<typename T>
inline void Swap(T &x, T &y)
{
	auto tmp = x;
	x = y;
	y = tmp;
}


inline int SmallestMultipleToFit(int at_least, int multiple_of)
{
	int result = (at_least + multiple_of - 1) / multiple_of;
	return result;
}

inline int CeilDiv(int x, int y)
{
	int result = x / y + int(x % y != 0);
	return result;
}

#define Assert(x) assert(x)