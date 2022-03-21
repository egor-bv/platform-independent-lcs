#pragma once
#include <string>
#include <map>
#include <vector>

struct ArgumentValue
{
	std::string str;
	int intValue;
};

class ArgumentList
{
	std::string calledWith;
	std::vector<ArgumentValue> positionalArgs;
	std::map<std::string, ArgumentValue> namedArgs;

public:
	ArgumentList(int argc, char **argv)
	{
		calledWith = std::string(argv[0]);
		for (int i = 1; i <= argc; ++argc)
		{
			ArgumentValue v;
		}
	}

	bool HasKey(std::string key)
	{
		return (bool)namedArgs.count(key);
	}

	ArgumentValue GetKey(std::string key)
	{
		return namedArgs[key];
	}

};
