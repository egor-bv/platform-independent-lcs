#pragma once
#include <CL/sycl.hpp>

#define REQD_SUB_GROUP_SIZE(SG_SIZE) [[intel::reqd_sub_group_size(SG_SIZE)]]



// Extra helpful stuff...
#include <sycl/ext/oneapi/experimental/builtins.hpp>


#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

// static const CONSTANT char FMT_GROUP[] = "group_id: %d, left_border: %d, right_border: %d\n";
// static const CONSTANT char FMT_GROUP[]


#define K_PRINTF(FMT, ...) do { const CONSTANT char FMT_[] = FMT; sycl::ext::oneapi::experimental::printf(FMT_, __VA_ARGS__); }while(0)
#define K_ASSERT(cond) if (!(cond)) { K_PRINTF("Assert failed '%s' %s:%d\n", #cond, __FILE__, __LINE__); }
#define K_ASSERT_INT_EQ(v0, v1) if ((v0) != (v1)) { K_PRINTF("Assert failed [%d != %d] <%s == %s> '%s':%d\n", (v0), (v1), #v0, #v1, __FILE__, __LINE__); }
