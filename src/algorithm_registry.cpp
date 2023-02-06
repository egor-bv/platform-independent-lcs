#include "algorithm_registry.hpp"

sycl::queue *LcsAlgorithmRegistry::GetQueueForDeviceType(std::string device_type)
{
	if (device_type == "cpu")
	{
		if (!QUEUE_CPU)
		{
			QUEUE_CPU = new sycl::queue(sycl::cpu_selector());
		}
		return QUEUE_CPU;
	}
	if (device_type == "gpu")
	{
		if (!QUEUE_GPU)
		{
			QUEUE_GPU = new sycl::queue(sycl::gpu_selector());
		}
		return QUEUE_GPU;
	}
	return nullptr;
}

PermutationMatrix SemiLocalLcsImpl::operator()(const LcsInput &input)
{
	LcsContext ctx(queue);
	func(input, ctx);
	if (ctx.h_strands && ctx.v_strands)
	{
		return PermutationMatrix::FromStrands(ctx.h_strands, ctx.h_strands_size,
											  ctx.v_strands, ctx.v_strands_size);
	}
	else
	{
		return std::move(ctx.matrix);
	}
}

SemiLocalLcsImpl LcsAlgorithmRegistry::Get(std::string name, std::string device_type)
{
	if (!reg.count(name))
	{
		return SemiLocalLcsImpl{};
	}
	auto *fn = reg[name];
	auto *q(GetQueueForDeviceType(device_type));

	auto result = SemiLocalLcsImpl{};
	result.ok = true;
	result.queue = q;
	result.func = fn;

	return result;
}

#include "lcs_reference.hpp"
#include "lcs_antidiagonal.hpp"
#include "lcs_stripes.hpp"
#include "lcs_tiled.hpp"
#include "lcs_hybrid.hpp"

#define SEMI(name, fn) reg[name] = fn;
LcsAlgorithmRegistry::LcsAlgorithmRegistry()
{
	SEMI("reference", (Lcs_Semi_Reference));
	SEMI("antidiagonal_8", (Lcs_Semi_Antidiagonal_ST<8>));
	SEMI("antidiagonal_16", (Lcs_Semi_Antidiagonal_ST<16>));
	SEMI("stripes_8", (Lcs_Semi_Stripes_ST<8>));
	SEMI("stripes_16", (Lcs_Semi_Stripes_ST<16>));

	SEMI("tiled_st_8", (Lcs_Semi_Tiled_ST_NewScoping<8, 3, 3>));
	SEMI("tiled_st_16", (Lcs_Semi_Tiled_ST_NewScoping<16, 3, 3>));

	SEMI("tiled_mt_8", (Lcs_Semi_Tiled_MT_Correct<8, 4, 4, 16>));
	SEMI("tiled_mt_8_32", (Lcs_Semi_Tiled_MT_Correct<8,  5, 5, 32>));
	SEMI("tiled_mt_8_64", (Lcs_Semi_Tiled_MT_Correct<8,  5, 5, 64>));
	SEMI("tiled_mt_8_128", (Lcs_Semi_Tiled_MT_Correct<8, 5, 5, 128>));
	SEMI("tiled_mt_8_256", (Lcs_Semi_Tiled_MT_Correct<8, 5, 5, 256>));
	SEMI("tiled_mt_8_512", (Lcs_Semi_Tiled_MT_Correct<8, 5, 5, 512>));
	
	SEMI("tiled_mt_16", (Lcs_Semi_Tiled_MT_Correct<16, 3, 3, 16>));

	SEMI("hybrid_mt_8", (Lcs_Semi_Antidiagonal_Hybrid_MT<8, 3>));
	SEMI("hybrid_mt_16", (Lcs_Semi_Antidiagonal_Hybrid_MT<16, 3>));
	SEMI("hybrid_depth_0", (Lcs_Semi_Antidiagonal_Hybrid_MT<8, 0>));
	SEMI("hybrid_depth_1", (Lcs_Semi_Antidiagonal_Hybrid_MT<8, 1>));
	SEMI("hybrid_depth_2", (Lcs_Semi_Antidiagonal_Hybrid_MT<8, 2>));
	SEMI("hybrid_depth_3", (Lcs_Semi_Antidiagonal_Hybrid_MT<8, 3>));
	SEMI("hybrid_depth_4", (Lcs_Semi_Antidiagonal_Hybrid_MT<8, 4>));
	SEMI("hybrid_depth_5", (Lcs_Semi_Antidiagonal_Hybrid_MT<8, 5>));
	SEMI("hybrid_depth_6", (Lcs_Semi_Antidiagonal_Hybrid_MT<8, 6>));
}
