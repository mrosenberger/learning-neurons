// Utilities and calculations

var calculateNeuronOutput = function(inputs, weights) {
  var sum = 0;
  for (var i=0; i < inputs.length; i++) {
    sum += (weights[i] * inputs[i].getOutput());
  };
  return Math.tanh(sum);
};

var addLayerToInputs = function(layer, neuron) {
  for (var i=0; i < layer.length; i++) {
    neuron.addInputNeuron(layer[i], 0);
  }
};

var resetLayerCaches = function(layer) {
  for (var i=0; i < layer.length; i++) {
    layer[i].resetCachedOutput();
  }
};

// Neuron implementation

var Neuron = function(bias, biasWeight) {
	this.inputs = [bias];
  this.weights = [biasWeight];
  this.cachedOutput = undefined;
};

Neuron.prototype.addInputNeuron = function(neuron, initialWeight) {
  this.inputs.push(neuron);
  this.weights.push(initialWeight);
};

Neuron.prototype.getOutput = function() {
  if (this.value == undefined) this.cachedOutput = calculateNeuronOutput(inputs, weights);
  return this.cachedOutput;
};

Neuron.prototype.resetCachedOutput = function() {
  this.value = undefined;
};

// InputNeuron implementation

var InputNeuron = function(outputValue) {
  this.outputValue = outputValue;
};

InputNeuron.prototype.getOutput = function() {
  return this.outputValue;
};

// NeuronNetwork implementation

var NeuronNetwork = function(inputLayerWidth, outputLayerWidth, hiddenLayerWidth, numHiddenLayers) {
  this.inputLayer = [];
  this.generateInputLayer(inputLayerWidth);
  this.hiddenLayers = [];
  this.generateHiddenLayers(numHiddenLayers, hiddenLayerWidth);
  this.outputLayer = [];
  this.generateOutputLayer(outputLayerWidth);
};

NeuronNetwork.prototype.generateInputLayer = function(layerWidth) {
  for (var neuronIndex=0; neuronIndex < layerWidth; neuronIndex++) {
    this.inputLayer.push(new InputNeuron(1));
  }
};

NeuronNetwork.prototype.generateHiddenLayers= function(numHiddenLayers, layerWidth) {
  for (var layerIndex=0; layerIndex < numHiddenLayers; layerIndex++) {
    var currentLayer = [];
    this.hiddenLayers.push(currentLayer);
    for (var neuronIndex=0; neuronIndex < layerWidth; neuronIndex++) {
      var currentNeuron = new Neuron(1, 1);
      if (layerIndex == 0) {
        addLayerToInputs(this.inputLayer, currentNeuron);
      } else {
        addLayerToInputs(this.hiddenLayers[layerIndex-1], currentNeuron);
      }
      currentLayer.push(currentNeuron);
    }
  }
};

NeuronNetwork.prototype.generateOutputLayer = function(layerWidth) {
  var lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length-1];
  for (var neuronIndex=0; neuronIndex < layerWidth; neuronIndex++) {
    var currentNeuron = new Neuron(1, 1);
    addLayerToInputs(lastHiddenLayer, currentNeuron);
  }
  this.outputLayer.push(currentNeuron);

};


NeuronNetwork.prototype.resetCachedOutputs = function() {

  for (var i=0; i < hiddenLayers.length; i++) {
    resetLayerCaches(hiddenLayers[i]);
  }

};
