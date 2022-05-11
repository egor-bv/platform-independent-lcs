#pragma once

#include "lcs_types.hpp"

#include <map>
#include <string>
#include <vector>

#include "permutation.hpp"



enum class LcsSolverKind
{
	Prefix,
	BinaryPrefix,
	Semilocal
};

// Contains LCS implementation function and compatible command queue
class LcsSolver
{
	std::string name;
	LcsSolverKind kind;
	sycl::queue *queue = nullptr;
	bool validating = false;

	LcsFunction solve;
public:

	std::string Name()
	{
		return name;
	}

	PermutationMatrix Semilocal(const std::vector<int> &a, const std::vector<int> &b)
	{
		LcsContext ctx(queue);
		LcsInput input(a.data(), a.size(), b.data(), b.size());
		solve(input, ctx);
		if (validating)
		{
			//...
		}
		return PermutationMatrix::FromStrands(ctx.h_strands, ctx.h_strands_size, ctx.v_strands, ctx.v_strands_size);

	}
	PermutationMatrix Semilocal(const int *a_data, int a_size, const int *b_data, int b_size)
	{
		LcsContext ctx(queue);
		LcsInput input(a_data, a_size, b_data, b_size);
		solve(input, ctx);
		if (validating)
		{
			//...
		}
		return PermutationMatrix::FromStrands(ctx.h_strands, ctx.h_strands_size, ctx.v_strands, ctx.v_strands_size);
	}
	PermutationMatrix Semilocal(const LcsInput &input)
	{
		LcsContext ctx(queue);
		solve(input, ctx);
		if (validating)
		{
			// ...
		}
		return PermutationMatrix::FromStrands(ctx.h_strands, ctx.h_strands_size, ctx.v_strands, ctx.v_strands_size);
	}

	int Prefix(const std::vector<int> &a, const std::vector<int> &b)
	{
		LcsContext ctx(queue);
		LcsInput input(a.data(), a.size(), b.data(), b.size());
		solve(input, ctx);
		return ctx.llcs;
	}
	int Prefix(const int *a_data, int a_size, const int *b_data, int b_size)
	{
		LcsContext ctx(queue);
		LcsInput input(a_data, a_size, b_data, b_size);
		solve(input, ctx);
		return ctx.llcs;
	}
	int Prefix(const LcsInput &input)
	{
		LcsContext ctx(queue);
		solve(input, ctx);
		return ctx.llcs;
	}

	LcsSolver(LcsSolverKind kind, std::string name, LcsFunction solve)
		: kind(kind)
		, name(name)
		, solve(solve)
	{
	}

	void SetQueue(sycl::queue *q)
	{
		queue = q;
	}

	void ToggleValidation(bool validate)
	{
		validating = validate;
	}
};



// Maps names to LCS implementations, used for testing/documentation
class LcsRegistry
{
	std::map<std::string, LcsSolver> solvers;
	typedef void(*LcsRegistryInitializer)(LcsRegistry *);

public:

	std::map<std::string, LcsSolver> &Solvers()
	{
		return solvers;
	}

	LcsRegistry(LcsRegistryInitializer init)
	{
		init(this);
	}

	void AddSolver(LcsSolverKind kind, std::string name, LcsFunction f)
	{
		LcsSolver solver(kind, name, f);
		solvers.insert({ name, solver });
	}
};


LcsRegistry *get_global_lcs_registry();