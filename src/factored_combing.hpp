#pragma once


struct LcsProblem
{

};


template<typename Sequence, typename StrandList>
struct CombingAccessor
{
	Sequence A;
	Sequence B;
	StrandList H;
	StrandList V;

	template<typename Handler, typename BufSeq, typename BufStrands>
	static CombingAccessor FromBuffers(Handler h, BufSeq buf_a, BufSeq buf_b, BufStrands buf_h, BufStrands buf_v)
	{
		CombingAccessor result =
		{
			buf_a.get_access<sycl::access::mode::read>(h),
			buf_b.get_access<sycl::access::mode::read>(h),
			buf_h.get_access<sycl::access::mode::read_write>(h),
			buf_v.get_access<sycl::access::mode::read_write>(h),

		};
		return result;
	}
};



template<typename StrandIndex, int Size>
struct RegisterStrandCache
{
	static constexpr int Size = Size;
	StrandIndex S[Size];
};

template <typename RegisterStrandCache>
void LoadStrandCache(RegisterStrandCache cache, int i)
{
	StrandIndex cache.S[i];
}


template<typename CombingAccessor>
void CombingStep(CombingAccessor acc, int i, int j)
{
	auto a_sym = acc.A[i];
	auto b_sym = acc.B[j];
	auto h_strand = acc.H[i];
	auto v_strand = acc.V[j];

	bool sym_equal = a_sym == b_sym;
	bool has_crossing = h_strand > v_strand;
	bool need_swap = sym_equal || has_crossing;

	acc.H[i] = need_swap ? v_strand : h_strand;
	acc.V[j] = need_swap ? h_strand : v_strand;
}