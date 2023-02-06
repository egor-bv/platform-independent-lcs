#pragma once

#include "lcs_types.hpp"

#include <map>
#include <string>
#include <functional>

// typedef PermutationMatrix SemiLocalLcsImpl(const LcsInput &);

struct SemiLocalLcsImpl
{
	bool ok = false;
	sycl::queue *queue;
	LcsFunction *func;
	PermutationMatrix operator()(const LcsInput &);
};


struct LcsAlgorithmRegistry
{
	std::map<std::string, LcsFunction *> reg;

	sycl::queue *QUEUE_CPU = nullptr;
	sycl::queue *QUEUE_GPU = nullptr;

	sycl::queue *GetQueueForDeviceType(std::string device_type);
	SemiLocalLcsImpl Get(std::string name, std::string device_type);


	LcsAlgorithmRegistry();
};