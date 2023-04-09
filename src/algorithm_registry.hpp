#pragma once

#include "lcs_types.hpp"
#include "lcs_interface_internal.hpp"

#include <vector>
#include <map>
#include <string>
#include <functional>



// typedef PermutationMatrix SemiLocalLcsImpl(const LcsInput &);

struct SemiLocalLcsImpl
{
	bool ok = false;
	sycl::queue *queue;
	LcsFunction *func;
	LcsProcedure *proc;
	PermutationMatrix operator()(const LcsInput &);
};



struct LcsAlgorithmRegistry
{
	std::map<std::string, LcsFunction *> reg;
	std::map<std::string, LcsProcedure *> reg2;

	sycl::queue *QUEUE_CPU = nullptr;
	sycl::queue *QUEUE_GPU = nullptr;
	sycl::queue *QUEUE_FPGA_EMULATOR = nullptr;
	sycl::queue *QUEUE_FPGA = nullptr;

	sycl::queue *GetQueueForDeviceType(std::string device_type);
	SemiLocalLcsImpl Get(std::string name, std::string device_type);


	LcsAlgorithmRegistry();
	~LcsAlgorithmRegistry();

};