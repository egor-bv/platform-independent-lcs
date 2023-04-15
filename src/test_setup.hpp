#pragma once

#include <string>
#include <map>
#include <vector>


struct TestCaseOptions
{
	std::string algorithm;
	std::string device_type;

	std::string file_a;
	std::string file_b;

	std::string rng_a;
	std::string rng_b;
	int size_a;
	int size_b;
	int seed_a;
	int seed_b;

	int iterations;
};


struct ArgumentEntry
{

	std::string key;
	std::string value;

	bool is_int = false;
	bool has_value = false;

	int as_int = -1;

	std::string error;

	ArgumentEntry() = default;

	ArgumentEntry(std::string key_, std::string value_)
	{
		key = key_;
		value = value_;
	}

};

bool IsWhitespace(char c)
{
	return (c == ' ' ||
			c == '\t');
}

bool IsEol(char c)
{
	return (c == '\n' ||
			c == '\r' ||
			c == '\0');
}


bool IsValueCharacter(char c)
{
	return (('a' <= c && c <= 'z') ||
			('A' <= c && c <= 'Z') ||
			('0' <= c && c <= '9') ||
			(c == '_'));
}

bool IsDigit(char c)
{
	return ('0' <= c && c <= '9');
}

bool TryParseInt(const char *str, int *result)
{
	int sign = 1;
	int total = 0;
	const char *at = str;
	if (at[0] == '-')
	{
		sign *= -1;
		++at;
	}

	while (IsDigit(at[0]))
	{
		total = total * 10 + (at[0] - '0');
		++at;
	}
	if (at[0] != '\0')
	{
		return false;
	}
	else
	{
		*result = sign*total;
		return true;
	}

}

struct ArgumentList
{
	std::map<std::string, ArgumentEntry> dict;


	struct Token
	{
		const char *first;
		int length;
	};

	// Parser state
	const char *at;
	Token token = {};


	void SkipSpaces()
	{
		while (IsWhitespace(at[0]) && !IsEol(at[0]))
		{
			++at;
		}
	}

	void ParseWord()
	{
		token.first = at;
		while (at[0] != '#' && !IsWhitespace(at[0]) && !IsEol(at[0]) && at[0] != '=')
		{
			++at;
		}
		token.length = at - token.first;
	}

	ArgumentEntry MaybeGet(std::string key)
	{
		if (dict.count(key))
		{
			return dict[key];
		}
		else
		{
			return ArgumentEntry{};
		}
	}

	ArgumentEntry MaybeGetInt(std::string key, int default_int)
	{
		if (dict.count(key))
		{
			ArgumentEntry arg = dict[key];
			if (arg.is_int) 
			{
				return arg;
			}
		}

		{
			ArgumentEntry result;
			result.is_int = true;
			result.as_int = default_int;
			return result;
		}
	}

	ArgumentList(const char *line)
	{
		at = line;
		while (!IsEol(at[0]))
		{
			// if comment -- skip to end of line
			if (at[0] == '#')
			{
				while (!IsEol(at[0]) && at[0] != '\0') ++at;
				break;
			}

			SkipSpaces();
			ParseWord();
			std::string key = std::string(token.first, token.length);
			// Handle errors?

			SkipSpaces();
			if (at[0] == '=')
			{
				// OK
				++at;
				SkipSpaces();
				ParseWord();
				ArgumentEntry value = ArgumentEntry(key, std::string(token.first, token.length));
				value.is_int = TryParseInt(value.value.c_str(), &value.as_int);
				dict[key] = value;
			}
			else
			{
				// just skip word as meaningless
				continue;
			}

			SkipSpaces();
			// dict.insert(std::make_pair(key, value));
		}
		while (IsEol(at[0]) && at[0] != '\0')
		{
			// Skip remaining end of line characters for good measure
			++at;
		}


	}

	void DebugPrint()
	{
		printf("{\n");
		for (auto &kv : dict)
		{
			printf("  %s = %s ", kv.first.c_str(), kv.second.value.c_str());
			if (kv.second.is_int)
			{
				printf("as_int(%d) ", kv.second.as_int);
			}
			if (!kv.second.error.empty())
			{
				printf("ERROR: %s", kv.second.error.c_str());
			}
			printf("\n");
		}
		printf("}\n");
	}


	TestCaseOptions ParseOptions()
	{
		TestCaseOptions opts;
		opts.algorithm = MaybeGet("algorithm").value;
		opts.device_type = MaybeGet("device_type").value;
		opts.file_a = MaybeGet("file_a").value;
		opts.file_b = MaybeGet("file_b").value;
		opts.rng_a = MaybeGet("rng_a").value;
		opts.rng_b = MaybeGet("rng_b").value;
		opts.size_a = MaybeGetInt("size_a", 256).as_int;
		opts.size_b = MaybeGetInt("size_b", 256).as_int;
		opts.seed_a = MaybeGet("seed_a").as_int;
		opts.seed_b = MaybeGet("seed_b").as_int;

		opts.iterations = MaybeGetInt("iterations", 1).as_int;
		return opts;
	}
};

std::vector<ArgumentList> ParseEntireScript(const char *contents)
{
	std::vector<ArgumentList> lists;
	const char *at = contents;
	for (;;)
	{
		ArgumentList single_line(at);
		if (single_line.dict.size() == 0)
		{
			if (single_line.at[0] == '\0')
			{
				break;
			}
			else
			{
				continue;
			}
		}
		lists.push_back(single_line);
		at = single_line.at;

		single_line.DebugPrint();
		if (single_line.at[0] == '\0')
		{
			break;
		}
	}
	
	return lists;
}

struct TestCaseResult
{
	std::string algorithm;
	std::string device_type;
	std::string device;

	// std::string source_a;
	// std::string source_b;

	int size_a;
	int size_b;

	int64_t hash;
	double elapsed_ms;

	std::string error;
};

#include <inttypes.h>

struct TestResultWriter
{
	FILE *file;
	bool need_close = true;
	TestResultWriter(const char *filename)
	{
		if (filename)
		{
			file = fopen(filename, "w");
		}
		else
		{
			need_close = false;
			file = stdout;
		}
	}

	~TestResultWriter()
	{
		if (need_close)
		{
			fclose(file);
		}
	}

	void WriteCsvHeader()
	{
		fprintf(file, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
				"algorithm",
				"device_type",
				"device",

				"size_a",
				"size_b",

				"source_a",
				"source_b",

				"elapsed_ms",
				"hash",
				"speed");
	}

	void WriteLine(const TestCaseOptions &opts, const TestCaseResult &res)
	{
		fprintf(file, "%s,", res.algorithm.c_str());
		fprintf(file, "%s,", res.device_type.c_str());
		fprintf(file, "%s,", res.device.c_str());
		fprintf(file, "%d,%d,", res.size_a, res.size_b);
		if (!opts.file_a.empty())
		{
			fprintf(file, "%s,", opts.file_a.c_str());
		}
		else
		{
			fprintf(file, "%s(%d),", opts.rng_a.c_str(), opts.seed_a);
		}
		if (!opts.file_b.empty())
		{
			fprintf(file, "%s,", opts.file_b.c_str());
		}
		else
		{
			fprintf(file, "%s(%d),", opts.rng_b.c_str(), opts.seed_b);
		}
		fprintf(file, "%f,", res.elapsed_ms);
		fprintf(file, "%" PRId64 ",", res.hash);
		double speed = ((int64_t)res.size_a * (int64_t)res.size_b) / (res.elapsed_ms * 1000.0);
		fprintf(file, "%f", speed);
		fprintf(file, "\n");
	}
	
	void Flush()
	{
		fflush(file);
	}
};