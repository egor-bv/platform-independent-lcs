#pragma once

#include "semi_reference.hpp"
#include "semi_single_subgroup_antidiagonal.hpp"
#include "semi_single_subgroup_staircase.hpp"
#include "semi_base_structure.hpp"

#include "permutation.hpp"
#include "utility.hpp"


PermutationMatrix SemiLcs_Reference(const InputSequencePair &given)
{
	return SemiLcsBase_NoSycl(StickyBraidComb_Reference, given, true);
}

PermutationMatrix SemiLcs_SubgroupAntidiagonal8(sycl::queue &q, const InputSequencePair &given)
{
	return SemiLcsBase_Sycl(StickyBraidComb_Antidiagonal<3>, q, given, true);
}

PermutationMatrix SemiLcs_SubgroupAntidiagonal16(sycl::queue &q, const InputSequencePair &given)
{
	return SemiLcsBase_Sycl(StickyBraidComb_Antidiagonal<4>, q, given, true);
}

PermutationMatrix SemiLcs_SubgroupStaircaseGlobal8(sycl::queue &q, const InputSequencePair &given)
{
	return SemiLcsBase_Sycl(StickyBraidComb_StaircaseGlobal<3>, q, given, true);
}

PermutationMatrix SemiLcs_SubgroupStaircaseGlobal16(sycl::queue &q, const InputSequencePair &given)
{
	return SemiLcsBase_Sycl(StickyBraidComb_StaircaseGlobal<4>, q, given, true);
}

PermutationMatrix SemiLcs_SubgroupStaircaseCrosslane(sycl::queue &q, const InputSequencePair &given)
{
	return SemiLcsBase_Sycl(StickyBraidComb_StaircaseCrosslane, q, given, true);
}

