#pragma once

#if 1
#define Min(X, Y) ((X) < (Y) ? (X) : (Y))
#define Max(X, Y) ((X) > (Y) ? (X) : (Y))
#define Clamp(X, E0, E1) Max(E0, Min(E1, X))
#else
#define Min(X, Y) std::min((X), (Y))
#define Max(X, Y) std::max((X), (Y))
#define Clamp(X, E0, E1) std::max(std::min((X), (E1)), (E0))
#endif


inline int SmallestMultipleToFit(int multiple_of, int to_fit)
{
	int result = (to_fit + multiple_of - 1) / multiple_of;
	return result;
}
