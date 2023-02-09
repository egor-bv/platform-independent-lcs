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
#define TILED_ST(SG_SIZE, TILE_M, TILE_N) \
SEMI("tiled_st_" #SG_SIZE "_" #TILE_M "_" #TILE_N, (Lcs_Semi_Tiled_ST_NewScoping<SG_SIZE, TILE_M, TILE_N>))

#define TILED_MT(SG_SIZE, TILE_M, TILE_N, SUBDIVISIONS) \
SEMI("tiled_mt_" #SG_SIZE "_" #TILE_M "_" #TILE_N "_" #SUBDIVISIONS, (Lcs_Semi_Tiled_MT_Correct<SG_SIZE, TILE_M, TILE_N, SUBDIVISIONS>))

#define HYBRID(SG_SIZE, DEPTH) \
SEMI("hybrid_" #SG_SIZE "_" #DEPTH, (Lcs_Semi_Antidiagonal_Hybrid<SG_SIZE, DEPTH>))

#define TILED_ST_PERMUTATIONS 0
#define TILED_MT_PERMUTATIONS 1

LcsAlgorithmRegistry::LcsAlgorithmRegistry()
{
	#if TILED_ST_PERMUTATIONS
	TILED_ST(8, 1, 1);
	TILED_ST(16, 1, 1);
	TILED_ST(8, 1, 2);
	TILED_ST(16, 1, 2);
	TILED_ST(8, 1, 3);
	TILED_ST(16, 1, 3);
	TILED_ST(8, 1, 4);
	TILED_ST(16, 1, 4);
	TILED_ST(8, 1, 5);
	TILED_ST(16, 1, 5);
	TILED_ST(8, 1, 6);
	TILED_ST(16, 1, 6);
	TILED_ST(8, 1, 7);
	TILED_ST(16, 1, 7);
	TILED_ST(8, 1, 8);
	TILED_ST(16, 1, 8);
	TILED_ST(8, 2, 1);
	TILED_ST(16, 2, 1);
	TILED_ST(8, 2, 2);
	TILED_ST(16, 2, 2);
	TILED_ST(8, 2, 3);
	TILED_ST(16, 2, 3);
	TILED_ST(8, 2, 4);
	TILED_ST(16, 2, 4);
	TILED_ST(8, 2, 5);
	TILED_ST(16, 2, 5);
	TILED_ST(8, 2, 6);
	TILED_ST(16, 2, 6);
	TILED_ST(8, 2, 7);
	TILED_ST(16, 2, 7);
	TILED_ST(8, 2, 8);
	TILED_ST(16, 2, 8);
	TILED_ST(8, 3, 1);
	TILED_ST(16, 3, 1);
	TILED_ST(8, 3, 2);
	TILED_ST(16, 3, 2);
	TILED_ST(8, 3, 3);
	TILED_ST(16, 3, 3);
	TILED_ST(8, 3, 4);
	TILED_ST(16, 3, 4);
	TILED_ST(8, 3, 5);
	TILED_ST(16, 3, 5);
	TILED_ST(8, 3, 6);
	TILED_ST(16, 3, 6);
	TILED_ST(8, 3, 7);
	TILED_ST(16, 3, 7);
	TILED_ST(8, 3, 8);
	TILED_ST(16, 3, 8);
	TILED_ST(8, 4, 1);
	TILED_ST(16, 4, 1);
	TILED_ST(8, 4, 2);
	TILED_ST(16, 4, 2);
	TILED_ST(8, 4, 3);
	TILED_ST(16, 4, 3);
	TILED_ST(8, 4, 4);
	TILED_ST(16, 4, 4);
	TILED_ST(8, 4, 5);
	TILED_ST(16, 4, 5);
	TILED_ST(8, 4, 6);
	TILED_ST(16, 4, 6);
	TILED_ST(8, 4, 7);
	TILED_ST(16, 4, 7);
	TILED_ST(8, 4, 8);
	TILED_ST(16, 4, 8);
	TILED_ST(8, 5, 1);
	TILED_ST(16, 5, 1);
	TILED_ST(8, 5, 2);
	TILED_ST(16, 5, 2);
	TILED_ST(8, 5, 3);
	TILED_ST(16, 5, 3);
	TILED_ST(8, 5, 4);
	TILED_ST(16, 5, 4);
	TILED_ST(8, 5, 5);
	TILED_ST(16, 5, 5);
	TILED_ST(8, 5, 6);
	TILED_ST(16, 5, 6);
	TILED_ST(8, 5, 7);
	TILED_ST(16, 5, 7);
	TILED_ST(8, 5, 8);
	TILED_ST(16, 5, 8);
	TILED_ST(8, 6, 1);
	TILED_ST(16, 6, 1);
	TILED_ST(8, 6, 2);
	TILED_ST(16, 6, 2);
	TILED_ST(8, 6, 3);
	TILED_ST(16, 6, 3);
	TILED_ST(8, 6, 4);
	TILED_ST(16, 6, 4);
	TILED_ST(8, 6, 5);
	TILED_ST(16, 6, 5);
	TILED_ST(8, 6, 6);
	TILED_ST(16, 6, 6);
	TILED_ST(8, 6, 7);
	TILED_ST(16, 6, 7);
	TILED_ST(8, 6, 8);
	TILED_ST(16, 6, 8);
	TILED_ST(8, 7, 1);
	TILED_ST(16, 7, 1);
	TILED_ST(8, 7, 2);
	TILED_ST(16, 7, 2);
	TILED_ST(8, 7, 3);
	TILED_ST(16, 7, 3);
	TILED_ST(8, 7, 4);
	TILED_ST(16, 7, 4);
	TILED_ST(8, 7, 5);
	TILED_ST(16, 7, 5);
	TILED_ST(8, 7, 6);
	TILED_ST(16, 7, 6);
	TILED_ST(8, 7, 7);
	TILED_ST(16, 7, 7);
	TILED_ST(8, 7, 8);
	TILED_ST(16, 7, 8);
	TILED_ST(8, 8, 1);
	TILED_ST(16, 8, 1);
	TILED_ST(8, 8, 2);
	TILED_ST(16, 8, 2);
	TILED_ST(8, 8, 3);
	TILED_ST(16, 8, 3);
	TILED_ST(8, 8, 4);
	TILED_ST(16, 8, 4);
	TILED_ST(8, 8, 5);
	TILED_ST(16, 8, 5);
	TILED_ST(8, 8, 6);
	TILED_ST(16, 8, 6);
	TILED_ST(8, 8, 7);
	TILED_ST(16, 8, 7);
	TILED_ST(8, 8, 8);
	TILED_ST(16, 8, 8);
	#endif
	
	#if TILED_MT_PERMUTATIONS
	TILED_MT(16, 4, 6, 4);
	TILED_MT(16, 4, 6, 8);
	TILED_MT(16, 4, 6, 16);
	TILED_MT(16, 4, 6, 32);
	TILED_MT(16, 4, 6, 64);
	TILED_MT(16, 4, 6, 128);
	TILED_MT(16, 4, 6, 256);
	TILED_MT(16, 4, 6, 512);
	
	
	TILED_MT(16, 2, 2, 128)
	TILED_MT(16, 2, 2, 256)
	TILED_MT(16, 2, 3, 128)
	TILED_MT(16, 2, 3, 256)
	TILED_MT(16, 2, 4, 128)
	TILED_MT(16, 2, 4, 256)
	TILED_MT(16, 2, 5, 128)
	TILED_MT(16, 2, 5, 256)
	TILED_MT(16, 2, 6, 128)
	TILED_MT(16, 2, 6, 256)
	TILED_MT(16, 3, 2, 128)
	TILED_MT(16, 3, 2, 256)
	TILED_MT(16, 3, 3, 128)
	TILED_MT(16, 3, 3, 256)
	TILED_MT(16, 3, 4, 128)
	TILED_MT(16, 3, 4, 256)
	TILED_MT(16, 3, 5, 128)
	TILED_MT(16, 3, 5, 256)
	TILED_MT(16, 3, 6, 128)
	TILED_MT(16, 3, 6, 256)
	TILED_MT(16, 4, 2, 128)
	TILED_MT(16, 4, 2, 256)
	TILED_MT(16, 4, 3, 128)
	TILED_MT(16, 4, 3, 256)
	TILED_MT(16, 4, 4, 128)
	TILED_MT(16, 4, 4, 256)
	TILED_MT(16, 4, 5, 128)
	TILED_MT(16, 4, 5, 256)
	TILED_MT(16, 4, 6, 128)
	TILED_MT(16, 4, 6, 256)
	TILED_MT(16, 5, 2, 128)
	TILED_MT(16, 5, 2, 256)
	TILED_MT(16, 5, 3, 128)
	TILED_MT(16, 5, 3, 256)
	TILED_MT(16, 5, 4, 128)
	TILED_MT(16, 5, 4, 256)
	TILED_MT(16, 5, 5, 128)
	TILED_MT(16, 5, 5, 256)
	TILED_MT(16, 5, 6, 128)
	TILED_MT(16, 5, 6, 256)
	TILED_MT(16, 6, 2, 128)
	TILED_MT(16, 6, 2, 256)
	TILED_MT(16, 6, 3, 128)
	TILED_MT(16, 6, 3, 256)
	TILED_MT(16, 6, 4, 128)
	TILED_MT(16, 6, 4, 256)
	TILED_MT(16, 6, 5, 128)
	TILED_MT(16, 6, 5, 256)
	TILED_MT(16, 6, 6, 128)
	TILED_MT(16, 6, 6, 256)
	#endif
}
