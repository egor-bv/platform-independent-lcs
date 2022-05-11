// corresponds to compilation unit, containing all LCS implementations

#include "lcs_registry.hpp"

static LcsRegistry *GLOBAL_LCS_REGISTRY = nullptr;

#if FPGA
// #include ...
// only compile FPGA-compatible items here...

void init_fpga(LcsRegistry *r)
{
	#define SEMI(name, f) r->AddSolver(LcsSolverKind::Semilocal, name, f)
	// ...
	#undef SEMI

	#define BINP(name, f) r->AddSolver(LcsSolverKind::BinaryPrefix, name, f)
	// ...
	#undef BINP
}

LcsRegistry *get_global_lcs_registry()
{
	if (!GLOBAL_LCS_REGISTRY)
	{
		LcsRegistry *r = new LcsRegistry(init_fpga);
		GLOBAL_LCS_REGISTRY = r;
	}
	return GLOBAL_LCS_REGISTRY;
}


#else
// #include ...
// only compile XPU-compatible items here...

#include "lcs_reference.hpp"
#include "lcs_antidiagonal.hpp"
#include "lcs_tiled.hpp"

void init_xpu(LcsRegistry *r)
{
	#define SEMI(name, f) r->AddSolver(LcsSolverKind::Semilocal, (name), f)
	SEMI("antidiagonal_st_8", Lcs_Semi_Antidiagonal_ST<8>);
	SEMI("antidiagonal_st_16", Lcs_Semi_Antidiagonal_ST<16>);
	SEMI("reference", Lcs_Semi_Reference);
	// SEMI("tiled_st_8_4_4", (Lcs_Semi_Tiled_ST<8, 4, 4>));
	// SEMI("tiled_st_16_4_4", (Lcs_Semi_Tiled_ST<16, 4, 4>));
	SEMI("tiled_st_ref", (Lcs_Semi_Tiled_ST_Reference<16, 4, 8>));
	SEMI("tiled_st_test", (Lcs_Semi_Tiled_ST<16, 4, 8>));
	SEMI("tiled_mt_test", (Lcs_Semi_Tiled_MT<16, 4, 8>));
	#undef SEMI

	#define BINP(name, f) r->AddSolver(LcsSolverKind::BinaryPrefix, name, f)
	// ...
	#undef BINP
}

LcsRegistry *get_global_lcs_registry()
{
	if (!GLOBAL_LCS_REGISTRY)
	{
		LcsRegistry *r = new LcsRegistry(init_xpu);
		GLOBAL_LCS_REGISTRY = r;
	}
	return GLOBAL_LCS_REGISTRY;
}

#endif