// Utilities and calculations

var log = function(message) {
  console.log(message);
};

var sigmoid = function(n) {
    return 1/(1+Math.pow(Math.E, -n));
}

var calculateNeuronOutput = function(inputs, weights) {
  var sum = 0;
  for (var i=0; i < inputs.length; i++) {
    sum += (weights[i] * inputs[i].getOutput());
  };
  return sigmoid(sum);
  //return Math.tanh(sum);
};

var resetLayerCaches = function(layer) {
  _.each(layer, function(neuron) {
    neuron.resetCachedOutput();
  });
};

var unsetValue = undefined;

// Neuron implementation

var Neuron = function() {
	this.inputs = [];
  this.weights = [];
  this.cachedOutput = unsetValue;
};

Neuron.prototype.addInputNeuron = function(neuron, initialWeight) {
  this.inputs.push(neuron);
  this.weights.push(initialWeight);
};

Neuron.prototype.getOutput = function() {
  if (this.cachedOutput == unsetValue) this.cachedOutput = calculateNeuronOutput(this.inputs, this.weights);
  return this.cachedOutput;
};

Neuron.prototype.resetCachedOutput = function() {
  this.cachedOutput = unsetValue;
};

Neuron.prototype.setCachedOutput = function(value) {
  this.cachedOutput = value;
};

// NeuronNetwork implementation

var NeuronNetwork = function(inputLayerWidth, outputLayerWidth, hiddenLayerWidth, numHiddenLayers) {
  this.biasNeuron = new Neuron();
  this.biasNeuron.setCachedOutput(-1);
  this.neuronLayers = this.generateLayers(inputLayerWidth, outputLayerWidth, hiddenLayerWidth, numHiddenLayers, this.biasNeuron, 1);
};

NeuronNetwork.prototype.generateLayers= function(inputLayerWidth, outputLayerWidth, hiddenLayerWidth, numHiddenLayers, biasNeuron, biasNeuronInitialWeight) {
  var generateLayer = function(layerWidth) {
    log("Generating layer of width " + layerWidth);
    return _.map(_.range(layerWidth), function(val) {
      var neur = new Neuron();
      neur.addInputNeuron(biasNeuron, biasNeuronInitialWeight);
      return neur;
    });
  };

  var connectLayers = function(closerToInput, closerToOutput) {
    _.map(closerToInput, function(input) {
      _.map(closerToOutput, function(output) {
        output.addInputNeuron(input, Math.random()-0.5);
      });
    });
  };

  var res = [];
  res.push(generateLayer(inputLayerWidth));
  for (var i=0; i < numHiddenLayers; i++) {
    res.push(generateLayer(hiddenLayerWidth));
    connectLayers(res[res.length-2], res[res.length-1]);
  }
  res.push(generateLayer(outputLayerWidth));
  connectLayers(res[res.length-2], res[res.length-1]);
  return res;
};

NeuronNetwork.prototype.resetCachedOutputs = function() {
  for (var i=0; i < this.neuronLayers.length; i++) {
    resetLayerCaches(this.neuronLayers[i]);
  }
};

NeuronNetwork.prototype.getOutputLayer = function() {
  return this.neuronLayers[this.neuronLayers.length-1];
};

NeuronNetwork.prototype.getInputLayer = function() {
  return this.neuronLayers[0];
};

NeuronNetwork.prototype.evaluate = function(inputValues) {
  var inputLayer = this.getInputLayer();
  if (inputValues.length != inputLayer.length) {
    throw "Inputs array must be the same length as input layer"
  }
  this.resetCachedOutputs();
  for (var i=0; i < inputLayer.length; i++) {
    inputLayer[i].setCachedOutput(inputValues[i]);
  }
  return _.map(this.getOutputLayer(), function(neuron) {
    return neuron.getOutput();
  });
};

