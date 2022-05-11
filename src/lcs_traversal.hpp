#pragma once

#include <CL/sycl.hpp>


template<typename Symbols, typename Strands, typename Action>
void TraverseAntidiagonal(Symbols as, Symbols bs, Strands hs, Strands vs, int m, int n, int diag_idx, sycl::nd_item<1> item)
{
	// length and stuff...
	int i_first = ;
	int j_first = ;
}