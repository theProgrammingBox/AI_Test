#include "Model.h"

/*
class Model
{
public:
	Model(int numInputs, int numHiddenNodes, int numOutputs);
	~Model();
	Model(const Model& other);
	Model(Model&& other) noexcept;
	Model& operator=(const Model& other);
	Model& operator=(Model&& other) noexcept;
	void forwardPropagate(const float* input, float* output);

private:
	int numInputs;
	int numHiddenNodes;
	int numOutputs;
	float* InToHidWeights;
	float* HidToHidWeights;
	float* HidToOutWeights;
	float* hiddenBiases;
	float* outputBiases;
	float* initialHidValues;
	vector<float*> hiddenValuesThruTime;
	vector<float*> inputValuesThruTime;
	vector<float*> outputValuesThruTime;

	void initialize();
	float leakyRelu(float x);
};
*/

Model::Model(int numInputs, int numHiddenNodes, int numOutputs)
{
	this->numInputs = numInputs;
	this->numHiddenNodes = numHiddenNodes;
	this->numOutputs = numOutputs;
	this->InToHidWeights = new float[numInputs * numHiddenNodes];
	this->HidToHidWeights = new float[numHiddenNodes * numHiddenNodes];
	this->HidToOutWeights = new float[numHiddenNodes * numOutputs];
	this->hiddenBiases = new float[numHiddenNodes];
	this->outputBiases = new float[numOutputs];
	this->initialHidValues = new float[numHiddenNodes];
	initialize();
}

Model::~Model()
{
	delete[] InToHidWeights;
	delete[] HidToHidWeights;
	delete[] HidToOutWeights;
	delete[] hiddenBiases;
	delete[] outputBiases;
	delete[] initialHidValues;
	for (int i = 0; i < hiddenValuesThruTime.size(); i++)
	{
		delete[] hiddenValuesThruTime[i];
	}
	for (int i = 0; i < inputValuesThruTime.size(); i++)
	{
		delete[] inputValuesThruTime[i];
	}
	for (int i = 0; i < outputValuesThruTime.size(); i++)
	{
		delete[] outputValuesThruTime[i];
	}
}

Model::Model(const Model& other)
{
	this->numInputs = other.numInputs;
	this->numHiddenNodes = other.numHiddenNodes;
	this->numOutputs = other.numOutputs;
	this->InToHidWeights = new float[numInputs * numHiddenNodes];
	this->HidToHidWeights = new float[numHiddenNodes * numHiddenNodes];
	this->HidToOutWeights = new float[numHiddenNodes * numOutputs];
	this->hiddenBiases = new float[numHiddenNodes];
	this->outputBiases = new float[numOutputs];
	this->initialHidValues = new float[numHiddenNodes];
	for (int i = 0; i < other.hiddenValuesThruTime.size(); i++)
	{
		this->hiddenValuesThruTime.push_back(new float[numHiddenNodes]);
	}
	for (int i = 0; i < other.inputValuesThruTime.size(); i++)
	{
		this->inputValuesThruTime.push_back(new float[numInputs]);
	}
	for (int i = 0; i < other.outputValuesThruTime.size(); i++)
	{
		this->outputValuesThruTime.push_back(new float[numOutputs]);
	}

	memcpy(this->InToHidWeights, other.InToHidWeights, numInputs * numHiddenNodes * sizeof(float));
	memcpy(this->HidToHidWeights, other.HidToHidWeights, numHiddenNodes * numHiddenNodes * sizeof(float));
	memcpy(this->HidToOutWeights, other.HidToOutWeights, numHiddenNodes * numOutputs * sizeof(float));
	memcpy(this->hiddenBiases, other.hiddenBiases, numHiddenNodes * sizeof(float));
	memcpy(this->outputBiases, other.outputBiases, numOutputs * sizeof(float));
	memcpy(this->initialHidValues, other.initialHidValues, numHiddenNodes * sizeof(float));
	for (int i = 0; i < other.hiddenValuesThruTime.size(); i++)
	{
		memcpy(this->hiddenValuesThruTime[i], other.hiddenValuesThruTime[i], numHiddenNodes * sizeof(float));
	}
	for (int i = 0; i < other.inputValuesThruTime.size(); i++)
	{
		memcpy(this->inputValuesThruTime[i], other.inputValuesThruTime[i], numInputs * sizeof(float));
	}
	for (int i = 0; i < other.outputValuesThruTime.size(); i++)
	{
		memcpy(this->outputValuesThruTime[i], other.outputValuesThruTime[i], numOutputs * sizeof(float));
	}
}

Model::Model(Model&& other) noexcept
{
	this->numInputs = other.numInputs;
	this->numHiddenNodes = other.numHiddenNodes;
	this->numOutputs = other.numOutputs;
	this->InToHidWeights = other.InToHidWeights;
	this->HidToHidWeights = other.HidToHidWeights;
	this->HidToOutWeights = other.HidToOutWeights;
	this->hiddenBiases = other.hiddenBiases;
	this->outputBiases = other.outputBiases;
	this->initialHidValues = other.initialHidValues;
	this->hiddenValuesThruTime = other.hiddenValuesThruTime;
	this->inputValuesThruTime = other.inputValuesThruTime;
	this->outputValuesThruTime = other.outputValuesThruTime;

	other.InToHidWeights = nullptr;
	other.HidToHidWeights = nullptr;
	other.HidToOutWeights = nullptr;
	other.hiddenBiases = nullptr;
	other.outputBiases = nullptr;
	other.initialHidValues = nullptr;
	for (int i = 0; i < other.hiddenValuesThruTime.size(); i++)
	{
		other.hiddenValuesThruTime[i] = nullptr;
	}
	for (int i = 0; i < other.inputValuesThruTime.size(); i++)
	{
		other.inputValuesThruTime[i] = nullptr;
	}
	for (int i = 0; i < other.outputValuesThruTime.size(); i++)
	{
		other.outputValuesThruTime[i] = nullptr;
	}
}

Model& Model::operator=(const Model& other)
{
	if (this != &other)
	{
		delete[] this->InToHidWeights;
		delete[] this->HidToHidWeights;
		delete[] this->HidToOutWeights;
		delete[] this->hiddenBiases;
		delete[] this->outputBiases;
		delete[] this->initialHidValues;
		for (int i = 0; i < this->hiddenValuesThruTime.size(); i++)
		{
			delete[] this->hiddenValuesThruTime[i];
		}
		for (int i = 0; i < this->inputValuesThruTime.size(); i++)
		{
			delete[] this->inputValuesThruTime[i];
		}
		for (int i = 0; i < this->outputValuesThruTime.size(); i++)
		{
			delete[] this->outputValuesThruTime[i];
		}
		this->hiddenValuesThruTime.clear();
		this->inputValuesThruTime.clear();
		this->outputValuesThruTime.clear();

		this->numInputs = other.numInputs;
		this->numHiddenNodes = other.numHiddenNodes;
		this->numOutputs = other.numOutputs;
		this->InToHidWeights = new float[numInputs * numHiddenNodes];
		this->HidToHidWeights = new float[numHiddenNodes * numHiddenNodes];
		this->HidToOutWeights = new float[numHiddenNodes * numOutputs];
		this->hiddenBiases = new float[numHiddenNodes];
		this->outputBiases = new float[numOutputs];
		this->initialHidValues = new float[numHiddenNodes];
		for (int i = 0; i < other.hiddenValuesThruTime.size(); i++)
		{
			this->hiddenValuesThruTime.push_back(new float[numHiddenNodes]);
		}
		for (int i = 0; i < other.inputValuesThruTime.size(); i++)
		{
			this->inputValuesThruTime.push_back(new float[numInputs]);
		}
		for (int i = 0; i < other.outputValuesThruTime.size(); i++)
		{
			this->outputValuesThruTime.push_back(new float[numOutputs]);
		}

		memcpy(this->InToHidWeights, other.InToHidWeights, numInputs * numHiddenNodes * sizeof(float));
		memcpy(this->HidToHidWeights, other.HidToHidWeights, numHiddenNodes * numHiddenNodes * sizeof(float));
		memcpy(this->HidToOutWeights, other.HidToOutWeights, numHiddenNodes * numOutputs * sizeof(float));
		memcpy(this->hiddenBiases, other.hiddenBiases, numHiddenNodes * sizeof(float));
		memcpy(this->outputBiases, other.outputBiases, numOutputs * sizeof(float));
		memcpy(this->initialHidValues, other.initialHidValues, numHiddenNodes * sizeof(float));
		for (int i = 0; i < other.hiddenValuesThruTime.size(); i++)
		{
			memcpy(this->hiddenValuesThruTime[i], other.hiddenValuesThruTime[i], numHiddenNodes * sizeof(float));
		}
		for (int i = 0; i < other.inputValuesThruTime.size(); i++)
		{
			memcpy(this->inputValuesThruTime[i], other.inputValuesThruTime[i], numInputs * sizeof(float));
		}
		for (int i = 0; i < other.outputValuesThruTime.size(); i++)
		{
			memcpy(this->outputValuesThruTime[i], other.outputValuesThruTime[i], numOutputs * sizeof(float));
		}
	}
	return *this;
}

Model& Model::operator=(Model&& other) noexcept
{
	if (this != &other)
	{
		delete[] this->InToHidWeights;
		delete[] this->HidToHidWeights;
		delete[] this->HidToOutWeights;
		delete[] this->hiddenBiases;
		delete[] this->outputBiases;
		delete[] this->initialHidValues;
		for (int i = 0; i < this->hiddenValuesThruTime.size(); i++)
		{
			delete[] this->hiddenValuesThruTime[i];
		}
		for (int i = 0; i < this->inputValuesThruTime.size(); i++)
		{
			delete[] this->inputValuesThruTime[i];
		}
		for (int i = 0; i < this->outputValuesThruTime.size(); i++)
		{
			delete[] this->outputValuesThruTime[i];
		}
		this->hiddenValuesThruTime.clear();
		this->inputValuesThruTime.clear();
		this->outputValuesThruTime.clear();

		this->numInputs = other.numInputs;
		this->numHiddenNodes = other.numHiddenNodes;
		this->numOutputs = other.numOutputs;
		this->InToHidWeights = other.InToHidWeights;
		this->HidToHidWeights = other.HidToHidWeights;
		this->HidToOutWeights = other.HidToOutWeights;
		this->hiddenBiases = other.hiddenBiases;
		this->outputBiases = other.outputBiases;
		this->initialHidValues = other.initialHidValues;
		this->hiddenValuesThruTime = other.hiddenValuesThruTime;

		other.InToHidWeights = nullptr;
		other.HidToHidWeights = nullptr;
		other.HidToOutWeights = nullptr;
		other.hiddenBiases = nullptr;
		other.outputBiases = nullptr;
		other.initialHidValues = nullptr;
		for (int i = 0; i < other.hiddenValuesThruTime.size(); i++)
		{
			other.hiddenValuesThruTime[i] = nullptr;
		}
		for (int i = 0; i < other.inputValuesThruTime.size(); i++)
		{
			other.inputValuesThruTime[i] = nullptr;
		}
		for (int i = 0; i < other.outputValuesThruTime.size(); i++)
		{
			other.outputValuesThruTime[i] = nullptr;
		}
	}
	return *this;
}

void Model::initialize()
{
	Random random;
	for (int i = 0; i < numInputs * numHiddenNodes; i++)
	{
		InToHidWeights[i] = random.DoubleRandom();
	}
	for (int i = 0; i < numHiddenNodes * numHiddenNodes; i++)
	{
		HidToHidWeights[i] = random.DoubleRandom();
	}
	for (int i = 0; i < numHiddenNodes * numOutputs; i++)
	{
		HidToOutWeights[i] = random.DoubleRandom();
	}
	for (int i = 0; i < numHiddenNodes; i++)
	{
		hiddenBiases[i] = random.DoubleRandom();
	}
	for (int i = 0; i < numOutputs; i++)
	{
		outputBiases[i] = random.DoubleRandom();
	}
	for (int i = 0; i < numHiddenNodes; i++)
	{
		initialHidValues[i] = random.DoubleRandom();
	}
}

float Model::leakyRelu(float x)
{
	if (x > 0)
	{
		return x;
	}
	return 0.01f * x;
}

void Model::forwardPropagate(float* input)
{
	if (hiddenValuesThruTime.size() == 0)
	{
		hiddenValuesThruTime.push_back(new float[numHiddenNodes]);
		memcpy(hiddenValuesThruTime[0], initialHidValues, numHiddenNodes * sizeof(float));
	}
	float* currentHiddenValues = hiddenValuesThruTime[hiddenValuesThruTime.size() - 1];
	float* futureHiddenValues = new float[numHiddenNodes];
	for (int i = 0; i < numHiddenNodes; i++)
	{
		float sum = hiddenBiases[i];
		int firstIndex = i * numInputs;
		for (int j = 0; j < numInputs; j++)
		{
			sum += input[j] * InToHidWeights[firstIndex + j];
		}
		firstIndex = i * numHiddenNodes;
		for (int j = 0; j < numHiddenNodes; j++)
		{
			sum += currentHiddenValues[j] * HidToHidWeights[firstIndex + j];
		}
		futureHiddenValues[i] += leakyRelu(sum);
	}
	hiddenValuesThruTime.push_back(futureHiddenValues);
	inputValuesThruTime.push_back(input);
	float* currentOutputValues = new float[numOutputs];
	for (int i = 0; i < numOutputs; i++)
	{
		float sum = outputBiases[i];
		int firstIndex = i * numHiddenNodes;
		for (int j = 0; j < numHiddenNodes; j++)
		{
			sum += futureHiddenValues[j] * HidToOutWeights[firstIndex + j];
		}
		currentOutputValues[i] = leakyRelu(sum);
	}
	outputValuesThruTime.push_back(currentOutputValues);
}