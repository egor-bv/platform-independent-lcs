#pragma once
#include <chrono>

class Stopwatch
{
private:
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point latest_time;
	bool stopped;

	void record_latest()
	{
		latest_time = std::chrono::high_resolution_clock::now();
	}

public:
	Stopwatch()
	{
		restart();
	}

	void restart()
	{
		stopped = false;
		start_time = std::chrono::high_resolution_clock::now();
	}

	void stop()
	{
		record_latest();
		stopped = true;
	}

	double elapsed_ms()
	{
		if (!stopped)
		{
			record_latest();
		}
		auto duration = latest_time - start_time;
		return std::chrono::duration<double, std::milli>(duration).count();
	}

};
