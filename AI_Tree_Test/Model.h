#pragma once
#include "Header.h"

using std::vector;

class Model
{
public:
	Model(int numInputs, int numHiddenNodes, int numOutputs);
	~Model();
	Model(const Model& other);
	Model(Model&& other) noexcept;
	Model& operator=(const Model& other);
	Model& operator=(Model&& other) noexcept;
	void forwardPropagate(float* input);

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