#include "algorithm_registry.hpp"


#include <sycl/ext/intel/fpga_extensions.hpp>

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
	if (device_type == "fpga_emulator")
	{
		if (!QUEUE_FPGA_EMULATOR)
		{
			QUEUE_FPGA_EMULATOR = new sycl::queue(sycl::ext::intel::fpga_emulator_selector());
		}
		return QUEUE_FPGA_EMULATOR;
	}
	if (device_type == "fpga")
	{
		if (!QUEUE_FPGA)
		{
			QUEUE_FPGA = new sycl::queue(sycl::ext::intel::fpga_selector());
		}
		return QUEUE_FPGA;
	}
	return nullptr;
}

PermutationMatrix SemiLocalLcsImpl::operator()(const LcsInput &input)
{
	if (func)
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
	if (proc)
	{
		LcsProblemContext pctx = {};
		pctx.a_given = (const Word *)input.a_data;
		pctx.a_length = input.a_size;
		pctx.b_given = (const Word *)input.b_data;
		pctx.b_length = input.b_size;
		pctx.queue = queue;
		proc(pctx);
		if (pctx.h_strands && pctx.v_strands)
		{
			return PermutationMatrix::FromStrands((int *)pctx.h_strands, pctx.h_strands_length,
												  (int *)pctx.v_strands, pctx.v_strands_length);
		}
		else return PermutationMatrix();

	}
}

SemiLocalLcsImpl LcsAlgorithmRegistry::Get(std::string name, std::string device_type)
{
	if (reg.count(name))
	{

		auto *fn = reg[name];
		auto *q(GetQueueForDeviceType(device_type));

		auto result = SemiLocalLcsImpl{};
		result.ok = true;
		result.queue = q;
		result.func = fn;

		return result;
	}
	if (reg2.count(name))
	{
		auto *proc = reg2[name];
		auto *q = GetQueueForDeviceType(device_type);

		auto result = SemiLocalLcsImpl{};
		result.ok = true;
		result.queue = q;

		result.proc = proc;

		return result;
	}
	return SemiLocalLcsImpl();
}



#include "lcs_reference.hpp"
#include "lcs_residual_fixup.hpp"

#include "lcs_tiled.hpp"

// #include "lcs_antidiagonal.hpp"
// #include "lcs_stripes.hpp"
// #include "lcs_tiled.hpp"
// #include "lcs_hybrid.hpp"
// #include "lcs_general.hpp"


#define SEMI(name, fn) reg[name] = fn;

#define GENERAL(SG_SIZE, TILE_M, TILE_N, DEPTH, SECTIONS) \
SEMI("g_" #SG_SIZE "_" #TILE_M "_" #TILE_N "_" #DEPTH "_" #SECTIONS, (Lcs_General<SG_SIZE, TILE_M, TILE_N, DEPTH, SECTIONS>))

#define TILED_ST(SG_SIZE, TILE_M, TILE_N) \
SEMI("tiled_st_" #SG_SIZE "_" #TILE_M "_" #TILE_N, (Lcs_Semi_Tiled_ST<SG_SIZE, TILE_M, TILE_N>))

#define TILED_MT(SG_SIZE, TILE_M, TILE_N, SECTIONS) \
SEMI("tiled_mt_" #SG_SIZE "_" #TILE_M "_" #TILE_N "_" #SECTIONS, (Lcs_Semi_Tiled_MT<SG_SIZE, TILE_M, TILE_N, SECTIONS>))

#define HYBRID(SG_SIZE, DEPTH) \
SEMI("hybrid_" #SG_SIZE "_" #DEPTH, (Lcs_Semi_Antidiagonal_Hybrid_MT<SG_SIZE, DEPTH>))

#define GENERAL_NAMED(name, SG_SIZE, TILE_M, TILE_N, DEPTH, SECTIONS) SEMI(name, (Lcs_General<SG_SIZE, TILE_M, TILE_N, DEPTH, SECTIONS>))
#define TILED_ST_NAMED(name, SG_SIZE, TILE_M, TILE_N) SEMI(name, (Lcs_Semi_Tiled_ST<SG_SIZE, TILE_M, TILE_N>))
#define TILED_MT_NAMED(name, ...) SEMI(name, (Lcs_Semi_Tiled_MT<__VA_ARGS__>))
#define HYBRID_NAMED(name, ...) SEMI(name, (Lcs_Semi_Antidiagonal_Hybrid_MT<__VA_ARGS__>))

#ifndef LCS_VARIANT_LIST_INCLUDE_FILE
#define LCS_USE_DEFAULT_VARIANTS 0
#endif


#define SEMI2(name, fn) reg2[name] = fn;

LcsAlgorithmRegistry::LcsAlgorithmRegistry()
{
	// Always include reference implementation
	SEMI("ref", (Lcs_Semi_Reference));
	SEMI2("ref2", (Lcs_Semi_Reference2));
	SEMI2("tiled", (Lcs_Semi_Tiled_ST<16, 3, 3>));
	SEMI2("mt", (Lcs_Semi_Tiled_MT<16, 5, 6, 16>));
	SEMI2("slm", (Lcs_Semi_Tiled_MT_SLM<16, 5, 6, 16>));
	// SEMI("aaa", (Lcs_Semi_Tiled_MT<8, 4, 4, 16>));
	// SEMI("slm", (Lcs_Semi_Tiled_SLM_MT<8, 4, 4, 16>));

	#if LCS_USE_DEFAULT_VARIANTS
	GENERAL(8, 4, 6, 2, 8);
	TILED_ST(8, 4, 6);
	TILED_MT(8, 4, 6, 8);
	HYBRID(8, 3);

	GENERAL_NAMED("general", 8, 3, 6, 2, 8);
	TILED_ST_NAMED("tiled_st", 8, 4, 6);
	TILED_MT_NAMED("tiled_mt", 8, 4, 6, 8);
	HYBRID_NAMED("hybrid", 8, 3);
	#endif

	#if defined(LCS_VARIANT_LIST_INCLUDE_FILE)
	#include LCS_VARIANT_LIST_INCLUDE_FILE
	#endif
}

LcsAlgorithmRegistry::~LcsAlgorithmRegistry()
{
	delete QUEUE_CPU;
	delete QUEUE_GPU;
	delete QUEUE_FPGA_EMULATOR;
	delete QUEUE_FPGA;
}
