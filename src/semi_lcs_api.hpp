#pragma once

#define SEMI_ANTIDIAGONAL 1
#define SEMI_STAIRCASE_GLOBAL 0
#define SEMI_STAIRCASE_CROSSLANE 0
#define SEMI_STAIRCASE_LOCAL 0
#define SEMI_TILED 1

#include "semi_reference.hpp"

#if SEMI_ANTIDIAGONAL 
#include "semi_single_subgroup_antidiagonal.hpp"
#endif
#if SEMI_STAIRCASE_GLOBAL || SEMI_STAIRCASE_CROSSLANE || SEMI_STAIRCASE_LOCAL
#include "semi_single_subgroup_staircase.hpp"
#endif
#include "semi_base_structure.hpp"

#include "semi_tiled.hpp"

#include "permutation.hpp"
#include "utility.hpp"

PermutationMatrix SemiLcs_NotImplemented()
{
	std::cout << "\n >>> Algorithm not implemented! <<< \n";
	return PermutationMatrix(1);
}

PermutationMatrix SemiLcs_Reference(const InputSequencePair &given)
{
	return SemiLcsBase_NoSycl(StickyBraidComb_Reference, given, true);
}

PermutationMatrix SemiLcs_SubgroupAntidiagonal8(sycl::queue &q, const InputSequencePair &given)
{
#if SEMI_ANTIDIAGONAL
	return SemiLcsBase_Sycl(StickyBraidComb_Antidiagonal<3>, q, given, true);
#else
	return SemiLcs_NotImplemented();
#endif
}

PermutationMatrix SemiLcs_SubgroupAntidiagonal16(sycl::queue &q, const InputSequencePair &given)
{
#if SEMI_ANTIDIAGONAL
	return SemiLcsBase_Sycl(StickyBraidComb_Antidiagonal<4>, q, given, true);
#else
	return SemiLcs_NotImplemented();
#endif
}

PermutationMatrix SemiLcs_SubgroupStaircaseGlobal8(sycl::queue &q, const InputSequencePair &given)
{
#if SEMI_STAIRCASE_GLOBAL
	return SemiLcsBase_Sycl(StickyBraidComb_StaircaseGlobal<3>, q, given, true);
#else
	return SemiLcs_NotImplemented();
#endif
}

PermutationMatrix SemiLcs_SubgroupStaircaseGlobal16(sycl::queue &q, const InputSequencePair &given)
{
#if SEMI_STAIRCASE_GLOBAL
	return SemiLcsBase_Sycl(StickyBraidComb_StaircaseGlobal<4>, q, given, true);
#else
	return SemiLcs_NotImplemented();
#endif
}

PermutationMatrix SemiLcs_SubgroupStaircaseCrosslane(sycl::queue &q, const InputSequencePair &given)
{
#if SEMI_STAIRCASE_CROSSLANE
	return SemiLcsBase_Sycl(StickyBraidComb_StaircaseCrosslane, q, given, true);
#else
	return SemiLcs_NotImplemented();
#endif
}

PermutationMatrix SemiLcs_SubgroupStaircaseLocal(sycl::queue &q, const InputSequencePair &given)
{
#if SEMI_STAIRCASE_LOCAL
	return SemiLcsBase_Sycl(StickyBraidComb_StaircaseLocal<3>, q, given, true);
#else
	return SemiLcs_NotImplemented();
#endif
}



