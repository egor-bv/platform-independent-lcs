#pragma once
#include <string>
#include <map>


// Because C++ standard library can't even do this
bool proper_string_to_int(const std::string &str, int *result_ptr)
{
	int value = 0;
	int i = 0;
	int sign = 1;

	if (str.length() == 0) return false;

	while (i < str.length() && str[i] == '-')
	{
		sign *= -1;
		i++;
	}
	while (i < str.length() && str[i] >= '0' && str[i] <= '9')
	{
		value *= 10;
		value += int(str[i]) - '0';
		i++;
	}
	if (i < str.length())
	{
		// Can't parse correctly
		return false;
	}
	else
	{
		*result_ptr = sign * value;
		return true;
	}
}


struct ArgumentEntry
{
	std::string key;
	std::string value;

	bool is_int = false;
	bool has_value = false;

	int int_value = -1;
};



struct CliArgumentParser
{
	std::map<std::string, ArgumentEntry> dict;

public:
	CliArgumentParser(int argc, const char **argv)
	{
		for (int i = 1; i <= argc; ++i)
		{
			if (argv[i])
			{
				printf("%d: %s\n", i, argv[i]);
				std::string word(argv[i]);
				size_t split = 0;
				while (word[split] != '=' && split < word.length())
				{
					split++;
				}
				ArgumentEntry arg;

				arg.key = word.substr(0, split);
				arg.value = split < word.length() ? word.substr(split + size_t(1)) : "";
				arg.has_value = !arg.value.empty();
				arg.int_value = 1;
				arg.is_int = proper_string_to_int(arg.value, &arg.int_value);

				dict[arg.key] = arg;
			}
		}
	}

	bool has_key(const std::string &key)
	{
		return dict.count(key) > 0;
	}

	ArgumentEntry get(std::string key)
	{
		return dict[key];
	}

	void opt_string(std::string &dst, const std::string &key)
	{
		if (has_key(key))
		{
			auto arg = get(key);
			if (arg.has_value)
			{
				dst = arg.value;
			}
		}
	}

	void opt_int(int &dst, const std::string &key)
	{
		if (has_key(key))
		{
			auto arg = get(key);
			if (arg.is_int)
			{
				dst = arg.int_value;
			}
		}
	}

	void opt_bool(bool &dst, const std::string &key)
	{
		if (has_key(key))
		{
			auto arg = get(key);
			if (arg.has_value && arg.is_int)
			{
				// Will only be false if written as key=0
				dst = bool(arg.int_value);
			}
			else
			{
				dst = true;
			}
		}
	}
};
