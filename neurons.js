// Vector class

var Vector = function(x, y) {
  this.x = x;
  this.y = y;
};

Vector.prototype.add = function(other) {
  return new Vector(this.x + other.x, this.y + other.y);
};

Vector.prototype.subtract = function(other) {
  return new Vector(this.x - other.x, this.y - other.y);
};

Vector.prototype.magnitude = function() {
  return Math.sqrt(Math.pow(this.x, 2.0) + Math.pow(this.y, 2.0));
};


Vector.prototype.scale = function(scalar) {
  return new Vector(this.x * scalar, this.y * scalar);
};

Vector.prototype.normalize = function() {
  var magnitude = this.magnitude();
  return new Vector(this.x / magnitude, this.y / magnitude);
}; 

// Utilities and calculations

var randomInRange = function(low, high) {
  return low + Math.random() * (high-low);
};

var log = function(message) {
  console.log(message);
};

var sigmoid = function(n) {
  return 1/(1+Math.pow(Math.E, -n));
};

var activation = function(n) {
  return sigmoid(n);
};

var activationPrime = function(n) {

};

var calculateNeuronOutput = function(inputs, weights) {
  var sum = 0;
  for (var i=0; i < inputs.length; i++) {
    sum += (weights[i] * inputs[i].getOutput());
  };
  return activation(sum);
};

var deltaRule = function(learningRate) {

};

var resetLayerCaches = function(layer) {
  _.each(layer, function(neuron) {
    neuron.resetCachedOutput();
  });
};

var calculateError = function(expectedValue, computedValue) {
  //return Math.abs(expectedValue - computedValue);
  return 0.5 * Math.pow(expectedValue - computedValue, 2);
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
  this.biasNeuron.setCachedOutput(1);
  this.neuronLayers = this.generateLayers(inputLayerWidth, outputLayerWidth, hiddenLayerWidth, numHiddenLayers, this.biasNeuron, 1);
  this.afterEvaluateCallback = function(network) {};
};

NeuronNetwork.prototype.generateLayers= function(inputLayerWidth, outputLayerWidth, hiddenLayerWidth, numHiddenLayers, biasNeuron, biasNeuronInitialWeight) {
  var generateLayer = function(layerWidth, addBiasNeuron) {
    log("Generating layer of width " + layerWidth);
    return _.map(_.range(layerWidth), function(val) {
      var neur = new Neuron();
      if (addBiasNeuron) neur.addInputNeuron(biasNeuron, biasNeuronInitialWeight);
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
  res.push(generateLayer(inputLayerWidth, false));
  for (var i=0; i < numHiddenLayers; i++) {
    res.push(generateLayer(hiddenLayerWidth, true));
    connectLayers(res[res.length-2], res[res.length-1]);
  }
  res.push(generateLayer(outputLayerWidth, true));
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

NeuronNetwork.prototype.getLayers = function() {
  return this.neuronLayers;
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
  this.afterEvaluateCallback(this);
  return _.map(this.getOutputLayer(), function(neuron) {
    return neuron.getOutput();
  });
};

NeuronNetwork.setAfterEvaluateCallback = function(callback) {
  this.afterEvaluateCallback = callback;
};

// trainingInputs and trainingOutputs are two arrays of arrays - each sub-array is
var DumbTrainer = function(network, trainingInputs, trainingOutputs) {
  if (network == undefined || network == null) throw "Parameter network must not be undefined or null";
  if (_.isEmpty(trainingInputs)) throw "Parameter trainingInputs must not be empty";
  if (_.isEmpty(trainingOutputs)) throw "Parameter trainingOutputs must not be empty";
  if (_.isEmpty(network.getInputLayer())) throw "Network input layer must not be empty";
  if (_.isEmpty(network.getOutputLayer())) throw "Network output layer must not be empty";
  if (network.getInputLayer().length !== trainingInputs[0].length) throw "Network input layer and training inputs array items must be of same length";
  if (network.getOutputLayer().length !== trainingOutputs[0].length) throw "Network ouptut layer and training outputs array items must be of same length";
  this.network = network;
  this.trainingInputs = trainingInputs;
  this.trainingOutputs = trainingOutputs;
};


DumbTrainer.prototype.calculateOutputError = function(outputNeurons, expectedValues) {
  if (outputNeurons.length != expectedValues.length) throw "Output neurons and expected values must be of same length"
  var error = 0;

  /*return _.map(
    _.zip(
      _.map(outputNeurons, function(outputNeuron) { return outputNeuron.getOutput(); }), 
      expectedValues), 
    function(pair) { return calculateError(pair[1], pair[0]); }
  );*/
  for (var i=0; i < outputNeurons.length; i++) {
    error += calculateError(expectedValues[i], outputNeurons[i].getOutput());
  }
  return error;
};

DumbTrainer.prototype.trainOnce = function(trainingInput, trainingOutput, learningRate) {
  var neuronLayers = this.network.getLayers();
  // We skip the input layer, layer 0, in the below loop:
  for (var layerIndex=neuronLayers.length-1; layerIndex > 0; layerIndex--) {
    var neuronLayer = neuronLayers[layerIndex];
    for (var neuronIndex=0; neuronIndex < neuronLayer.length; neuronIndex++) {
      var neuron = neuronLayer[neuronIndex];
      for (var inputIndex=0; inputIndex < neuron.inputs.length; inputIndex++) {
        this.network.evaluate(trainingInput);
        var initialError = this.calculateOutputError(this.network.getOutputLayer(), trainingOutput);
        //log("Initial error: " + initialError);
        neuron.weights[inputIndex] += learningRate;
        this.network.evaluate(trainingInput);
        var afterAddingError = this.calculateOutputError(this.network.getOutputLayer(), trainingOutput);
        if (afterAddingError > initialError) { // Adding didn't work, so let's instead subtract. We have to subtract twice the learningRate because we already added up above
          neuron.weights[inputIndex] -= (2*learningRate);
        }
      }
    };
  }
};

DumbTrainer.prototype.train = function(times) {
  for (var i=0; i < times; i++) {
    var index = Math.floor(Math.random() * this.trainingInputs.length);
    this.trainOnce(this.trainingInputs[index], this.trainingOutputs[index], 0.01);
  }
};

var NeuronNetworkRenderer = function(context, network) {
  this.context = context;
  this.network = network;
};

NeuronNetworkRenderer.prototype.calculateNeuronPositionPure = function(canvasWidth, canvasHeight, layerIndex, numLayers, neuronIndex, numNeurons, horizontalPadding, verticalPadding) {
  var computeNthPosition = function(n, outOf, pixels) {
    var intervalPixels = pixels / (outOf);
    return (n) * intervalPixels + (0.5*intervalPixels);
  };
  var computeNthPositionPadded = function(n, outOf, pixels, padding) {
    return padding + computeNthPosition(n, outOf, pixels-(padding*2));
  };
  var padding = 0;
  return {
    x: computeNthPositionPadded(neuronIndex, numNeurons, canvasWidth, horizontalPadding),
    y: computeNthPositionPadded(layerIndex, numLayers, canvasHeight, verticalPadding)
  };
};

NeuronNetworkRenderer.prototype.calculateNeuronPosition = function(layerIndex, neuronIndex) {
  return this.calculateNeuronPositionPure(
    this.context.canvas.width, 
    this.context.canvas.height, 
    layerIndex, 
    this.network.getLayers().length, 
    neuronIndex,
    this.network.getLayers()[layerIndex].length,
    20,
    0);
};

NeuronNetworkRenderer.prototype.update = function(showOutputs) {

  var fontName = "Arial";
  var layers = this.network.getLayers();
  var highestWeightMagnitude = -Infinity;
  var lowestWeightMagnitude = Infinity;
  for (var layerIndex=0; layerIndex < layers.length; layerIndex++) {
    var layer = layers[layerIndex];
    for (var neuronIndex=0; neuronIndex < layer.length; neuronIndex++) {
      var neuron = layer[neuronIndex];
      for (var weightIndex=1; weightIndex < neuron.weights.length; weightIndex++) {
        var absWeight = Math.abs(neuron.weights[weightIndex]);
        if (absWeight > highestWeightMagnitude) highestWeightMagnitude = absWeight;
        if (absWeight < lowestWeightMagnitude) lowestWeightMagnitude = absWeight;
      }
    }
  }

  var normalizeWeightForColoring = function(value) { // Normalizes using the above enclosed highest and lowest weights
    //return (value - lowestWeightMagnitude) / highestWeightMagnitude;
    var a1 = lowestWeightMagnitude;
    var a2 = highestWeightMagnitude;
    var b1 = 0.0;
    var b2 = 1.0;
    var s = Math.abs(value);
    return b1 + ((s-a1)*(b2-b1)) / (a2-a1);
  };

  var truncate = function(n, decimals) {
    return parseFloat(n).toFixed(decimals);
  };

  var inputWeightPosition = function(originCoordinates, destinationCoordinates, index) { // Index is just used to switch between even and odd for spacing of weights
    /*return { 
      x: Math.floor((originCoordinates.x + destinationCoordinates.x) / 2.0),
      y: Math.floor((originCoordinates.y + destinationCoordinates.y) / 2.0)
    };*/
    var originVector = new Vector(originCoordinates.x, originCoordinates.y);
    var destinationVector = new Vector(destinationCoordinates.x, destinationCoordinates.y);
    var originToDestination = originVector.subtract(destinationVector);
    //var scaled = originToDestination.scale(randomInRange(0.2, 0.3));
    var scaled = originToDestination.scale((index % 2 == 0) ? 0.15 : 0.10);
    var result = destinationVector.add(scaled);
    return {x: result.x, y: result.y};
  };

  this.context.fillStyle = "white";
  this.context.fillRect(0, 0, this.context.canvas.width, this.context.canvas.height);
  
  for (var layerIndex=0; layerIndex < layers.length; layerIndex++) {
    var layer = layers[layerIndex];
    for (var neuronIndex=0; neuronIndex < layer.length; neuronIndex++) {
      var neuron = layer[neuronIndex];
      var coordinates = this.calculateNeuronPosition(layerIndex, neuronIndex);
      if (layerIndex > 0) {
        for (var weightIndex=1; weightIndex < neuron.weights.length; weightIndex++) {
          var inputCoordinates = this.calculateNeuronPosition(layerIndex-1, weightIndex-1);
          var weight = neuron.weights[weightIndex];
          //this.context.strokeStyle = "#cccccc";
          this.context.strokeStyle = (weight >= 0) ? "green" : "red";
          this.context.lineWidth = normalizeWeightForColoring(weight) * 2 + 0.1;
          this.context.beginPath();
          this.context.moveTo(coordinates.x, coordinates.y);
          this.context.lineTo(inputCoordinates.x, inputCoordinates.y);
          this.context.closePath();
          this.context.stroke();

          this.context.fillStyle = "black";
          this.context.font = "8px " + fontName;
          var weightPos = inputWeightPosition(inputCoordinates, coordinates, weightIndex);
          this.context.fillText(truncate(weight, 2), weightPos.x-7, weightPos.y);
        }
      }

      var output = neuron.getOutput();
      this.context.fillStyle = "blue";
      this.context.font = "8px " + fontName;
      var showOutput = showOutputs && (output != unsetValue);
      var showBias = neuron.weights.length > 0;
      if (showOutput && showBias) {
        this.context.fillText("Bias Weight: " + truncate(neuron.weights[0], 2), coordinates.x+10, coordinates.y-3);
        this.context.fillText("Output: " + truncate(output, 2), coordinates.x+10, coordinates.y+10);
      } else if (showBias) {
        this.context.fillText("Bias Weight: " + truncate(neuron.weights[0], 2), coordinates.x+10, coordinates.y+3);
      } else if (showOutput) {
        this.context.fillText("Output: " + truncate(output, 2), coordinates.x+10, coordinates.y+3);
      }

      this.context.fillStyle = "black";
      //this.context.strokeStyle = "black";
      //this.context.lineWidth = 1;
      this.context.beginPath();
      this.context.arc(coordinates.x, coordinates.y, 5, 0, Math.PI*2, true); 
      this.context.closePath();
      //this.context.stroke();
      this.context.fill();

    }
  }
};

var network = new NeuronNetwork(5, 2, 4, 2);
//var swapTrainingInputs =  [[0, 1, 0], [1, 0, 0], [0, 0, 1]];
//var swapTrainingOutputs = [[0, 1, 0], [0, 0, 1], [1, 0, 0]];
//var andTrainingInputs =  [[0, 0], [0, 1], [1, 0], [1, 1]];
//var andTrainingOutputs = [[0], [0], [0], [1]];

var trainingSets = {
  xor: {
    inputs:  [[0, 0], [0, 1], [1, 0], [1, 1]],
    outputs: []
  },
  and: {
    inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
    outputs: [[0], [0], [0], [1]]
  }
};

var trainingSet = trainingSets.and;
//var trainer = new DumbTrainer(network, trainingSet.inputs, trainingSet.outputs);

// Canvas and context
var canvas = document.getElementById("neuron-canvas");
var context = canvas.getContext("2d");
// Give the canvas focus
//canvas.focus();

var renderer = new NeuronNetworkRenderer(context, network);
renderer.update(true);

/*window.setInterval(function() {
  trainer.train(1);
  renderer.update(true);
}, 1000);*/
