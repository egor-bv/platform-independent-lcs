#pragma once

#include "lcs_types.hpp"

#include <map>
#include <string>

enum class LcsAlgorithmKind
{
	Prefix,
	BinaryPrefix,
	SemiLocal
};

struct LcsAlgorithmEntry
{
	std::string name;
	LcsAlgorithmKind kind;
	LcsProcedure proc;
};


// maps name to an LCS implementation function, used for testing/documentation
class LcsRegistry
{
	std::map<std::string, LcsAlgorithmEntry> algos;
	typedef void(*LcsRegistryInitializer)(LcsRegistry *);

public:
	LcsRegistry(LcsRegistryInitializer init)
	{
		init(this);
	}
};


extern LcsRegistry GLOBAL_LCS_REGISTRY;
