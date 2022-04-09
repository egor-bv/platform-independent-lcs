// corresponds to compilation unit, containing all LCS implementations

#include "lcs_registry.hpp"


#if FPGA
// #include ...
// only compile FPGA-compatible items here...


#else
// #include ...
// only compile XPU-compatible items here...

LcsRegistry GLOBAL_LCS_REGISTRY([](LcsRegistry *REGISTRY)
	{

	}
);

#endif