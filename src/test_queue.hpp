#pragma once

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_device_selector.hpp>

class QueueCreator
{
	sycl::queue *queue_cpu;
	sycl::queue *queue_gpu;
	sycl::queue *queue_host;
	sycl::queue *queue_fpga;

public:
	// TODO: set exception handler during queue creation
	sycl::queue &GetOrCreateQueue(std::string device)
	{
		if (device == "cpu")
		{
			if (!queue_cpu)
			{
				queue_cpu = new sycl::queue(sycl::cpu_selector());
			}
			return *queue_cpu;
		}

		if (device == "gpu")
		{

			if (!queue_gpu)
			{
				queue_gpu = new sycl::queue(sycl::gpu_selector());
			}
			return *queue_gpu;
		}

		if (device == "host")
		{
			if (!queue_host)
			{
				queue_host = new sycl::queue(sycl::host_selector());
			}
			return *queue_host;
		}

		if (device == "fpga")
		{
			if (!queue_fpga)
			{
				queue_fpga = new sycl::queue(sycl::ext::intel::fpga_selector());
			}
			return *queue_fpga;
		}
	}

	~QueueCreator()
	{
		if (queue_cpu) delete queue_cpu;
		if (queue_gpu) delete queue_gpu;
		if (queue_host) delete queue_host;
		if (queue_fpga) delete queue_fpga;
	}
};

QueueCreator GLOBAL_QUEUE_CREATOR;

sycl::queue &GetOrCreateQueue(std::string device)
{
	GLOBAL_QUEUE_CREATOR.GetOrCreateQueue(device);
}