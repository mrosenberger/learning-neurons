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
  //return n > 0 ? 1 : 0;
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
  this.errorValue = unsetValue;
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

Neuron.prototype.getError = function() {
  return this.errorValue;
};

Neuron.prototype.setError = function(newError) {
  this.errorValue = newError;
};

// NeuronNetwork implementation

var NeuronNetwork = function(inputLayerWidth, outputLayerWidth, hiddenLayerWidth, numHiddenLayers) {
  if (inputLayerWidth < 1) throw "Parameter inputLayerWidth must be greater than 0";
  if (outputLayerWidth < 1) throw "Parameter outputLayerWidth must be greater than 0";
  if ((numHiddenLayers > 0) && (hiddenLayerWidth < 1)) throw "Cannot specify nonzero positive hidden layers but zero hidden layer width"
  this.biasNeuron = new Neuron();
  this.biasNeuron.setCachedOutput(1);
  this.neuronLayers = this.generateLayers(inputLayerWidth, outputLayerWidth, hiddenLayerWidth, numHiddenLayers, this.biasNeuron);
  this.afterEvaluateCallback = function() {};
};

NeuronNetwork.prototype.generateLayers= function(inputLayerWidth, outputLayerWidth, hiddenLayerWidth, numHiddenLayers, biasNeuron) {
  var generateLayer = function(layerWidth, addBiasNeuron) {
    log("Generating layer of width " + layerWidth);
    return _.map(_.range(layerWidth), function(val) {
      var neur = new Neuron();
      if (addBiasNeuron) neur.addInputNeuron(biasNeuron, randomInRange(-0.1, 0.1));
      return neur;
    });
  };

  var connectLayers = function(closerToInput, closerToOutput) {
    _.map(closerToInput, function(input) {
      _.map(closerToOutput, function(output) {
        output.addInputNeuron(input, randomInRange(-0.1, 0.1));
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
  this.afterEvaluateCallback();
  return _.map(this.getOutputLayer(), function(neuron) {
    return neuron.getOutput();
  });
};

NeuronNetwork.prototype.setAfterEvaluateCallback = function(callback) {
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
  if (network.getOutputLayer().length !== trainingOutputs[0].length) throw "Network output layer and training outputs array items must be of same length";
  this.network = network;
  this.trainingInputs = trainingInputs;
  this.trainingOutputs = trainingOutputs;
};

DumbTrainer.prototype.calculateOutputError = function(outputNeurons, expectedValues) {
  if (outputNeurons.length != expectedValues.length) throw "Output neurons and expected values must be of same length"
  var error = 0;
  for (var i=0; i < outputNeurons.length; i++) {
    error += Math.pow(expectedValues[i] - outputNeurons[i].getOutput(), 2);
  }
  return 0.5*error;
};

DumbTrainer.prototype.trainOnce = function(trainingInput, trainingOutput, learningRate) {
  var neuronLayers = this.network.getLayers();
  var totalError = 0;
  var totalUpdates = 0;
  // We skip the input layer, layer 0, in the below loop:
  for (var layerIndex=neuronLayers.length-1; layerIndex > 0; layerIndex--) {
    var neuronLayer = neuronLayers[layerIndex];
    for (var neuronIndex=0; neuronIndex < neuronLayer.length; neuronIndex++) {
      var neuron = neuronLayer[neuronIndex];
      for (var inputIndex=0; inputIndex < neuron.inputs.length; inputIndex++) {
        this.network.evaluate(trainingInput);
        var initialError = this.calculateOutputError(this.network.getOutputLayer(), trainingOutput);
        //log("Initial error: " + initialError);
        totalError += initialError;
        totalUpdates++;
        var addAmount = (Math.random() - 0.5) * learningRate / (1.0 / Math.abs(neuron.weights[inputIndex]));
        neuron.weights[inputIndex] += addAmount;
        this.network.evaluate(trainingInput);
        var afterAddingError = this.calculateOutputError(this.network.getOutputLayer(), trainingOutput);
        if (afterAddingError > initialError) { // Adding didn't work
          neuron.weights[inputIndex] -= addAmount;
          //neuron.weights[inputIndex] -= (2*learningRate);
        }
      }
    };
  }
  return totalError / totalUpdates;
};

DumbTrainer.prototype.train = function(times) {
  var totalError = 0;
  for (var i=0; i < times; i++) {
    var index = Math.floor(Math.random() * this.trainingInputs.length);
    totalError += this.trainOnce(this.trainingInputs[index], this.trainingOutputs[index], 0.01);
  }
  return totalError / times;
};

// trainingInputs and trainingOutputs are two arrays of arrays - each sub-array is
var BackPropagationTrainer = function(network, trainingInputs, trainingOutputs) {
  if (network == undefined || network == null) throw "Parameter network must not be undefined or null";
  if (_.isEmpty(trainingInputs)) throw "Parameter trainingInputs must not be empty";
  if (_.isEmpty(trainingOutputs)) throw "Parameter trainingOutputs must not be empty";
  if (_.isEmpty(network.getInputLayer())) throw "Network input layer must not be empty";
  if (_.isEmpty(network.getOutputLayer())) throw "Network output layer must not be empty";
  if (network.getInputLayer().length !== trainingInputs[0].length) throw "Network input layer and training inputs array items must be of same length";
  if (network.getOutputLayer().length !== trainingOutputs[0].length) throw "Network output layer and training outputs array items must be of same length";
  this.network = network;
  this.trainingInputs = trainingInputs;
  this.trainingOutputs = trainingOutputs;
  this.debugMode = false;
};

BackPropagationTrainer.prototype.trainOnce = function(trainingInput, trainingOutput, learningRate) {
  if (_.isEmpty(trainingInput)) throw "Parameter trainingInput cannot be empty, null, or undefined";
  if (_.isEmpty(trainingOutput)) throw "Parameter trainingOutput cannot be empty, null, or undefined";
  if (learningRate == null || learningRate == undefined) throw "Parameter learningRate cannot be undefined or null";

  this.network.evaluate(trainingInput);
  var layers = this.network.getLayers();
  for (var layerIndex=(layers.length-1); layerIndex > 0; layerIndex--) { // Skip layer 0
    var layer = layers[layerIndex];
    if (layerIndex === layers.length-1) { // Output layer
      if (this.debugMode) log("Training layer " + layerIndex + " (output layer) containing " + layer.length + " neurons");
      for (var neuronIndex=0; neuronIndex < layer.length; neuronIndex++) {

        // Compute initial values:
        var neuron = layer[neuronIndex];
        var targetValue = trainingOutput[neuronIndex];
        var actualValue = neuron.getOutput();
        if (this.debugMode) log("target: " + targetValue);
        if (this.debugMode) log("actual: " + actualValue);

        // Compute error based on derivative of activation function and difference between training value actual value:
        var error = actualValue * (1.0 - actualValue) * (targetValue - actualValue);

        // Store that value into the neuron for later use by layers above us:
        neuron.setError(error);

        // Update weights to the above layer:
        for (var inputIndex=0; inputIndex < neuron.inputs.length; inputIndex++) {
          var adjustment = learningRate * error * neuron.inputs[inputIndex].getOutput();
          neuron.weights[inputIndex] += adjustment;
        }

      }
    } else { // Not output layer
      if (this.debugMode) log("Training layer " + layerIndex + " containing " + layer.length + " neurons");
      for (var neuronIndex=0; neuronIndex < layer.length; neuronIndex++) {

        // Compute initial values:
        var neuron = layer[neuronIndex];
        var actualValue = neuron.getOutput();
        var error = actualValue * (1.0 - actualValue);
        var summedWeightsAndErrors = 0;
        var nextLayer = layers[layerIndex+1];

        // Collect errors multiplied by weights from the layer closer to the output from our current position:
        for (var nextLayerNeuronIndex=0; nextLayerNeuronIndex<nextLayer.length; nextLayerNeuronIndex++) {
          var nextLayerNeuron = nextLayer[nextLayerNeuronIndex];
          var nextLayerNeuronWeight = nextLayerNeuron.weights[neuronIndex+1];
          var nextLayerNeuronError = nextLayerNeuron.getError();
          summedWeightsAndErrors += (nextLayerNeuronWeight * nextLayerNeuronError);
        }

        // Multiply the previously computed error value by the summed values we just collected:
        error *= summedWeightsAndErrors;

        // Update weights to the above layer (currently works for input layer only):
        for (var inputIndex=0; inputIndex < neuron.inputs.length; inputIndex++) {
          var adjustment = learningRate * error * neuron.inputs[inputIndex].getOutput();
          neuron.weights[inputIndex] += adjustment;
        }

      }
    }
  }
};

BackPropagationTrainer.prototype.train = function(times, learningRate) {
  for (var i=0; i < times; i++) {
    var index = Math.floor(Math.random() * this.trainingInputs.length);
    this.trainOnce(this.trainingInputs[index], this.trainingOutputs[index], learningRate);
  }
};

var NeuronNetworkRenderer = function(context, network, config) {
  this.context = context;
  this.network = network;
  this.config = config;
};

// Compute the position of a neuron on the canvas, without using any free variables:
NeuronNetworkRenderer.prototype.calculateNeuronPositionPure = function(canvasWidth, canvasHeight, layerIndex, numLayers, neuronIndex, numNeurons, horizontalPadding, verticalPadding) {
  var computeNthPosition = function(n, outOf, pixels) {
    var intervalPixels = pixels / (outOf);
    return (n) * intervalPixels + (0.5*intervalPixels);
  };
  var computeNthPositionPadded = function(n, outOf, pixels, padding) {
    return padding + computeNthPosition(n, outOf, pixels-(padding*2));
  };
  var padding = 0;
  return new Vector(
    computeNthPositionPadded(neuronIndex, numNeurons, canvasWidth, horizontalPadding),
    computeNthPositionPadded(layerIndex, numLayers, canvasHeight, verticalPadding)
  );
};

// Calculate the position of a neuron onscreen, utilizing knowledge about member network:
NeuronNetworkRenderer.prototype.calculateNeuronPosition = function(layerIndex, neuronIndex) {
  return this.calculateNeuronPositionPure(
    this.context.canvas.width, 
    this.context.canvas.height, 
    layerIndex, 
    this.network.getLayers().length, 
    neuronIndex,
    this.network.getLayers()[layerIndex].length,
    this.config.horizontalPadding,
    this.config.verticalPadding);
};

 // Normalizes using the above enclosed highest and lowest weights:
NeuronNetworkRenderer.prototype.normalizeQuantityForColoring = function(value, lowestMagnitude, highestMagnitude) {
  var a1 = lowestMagnitude;
  var a2 = highestMagnitude;
  var b1 = 0.0;
  var b2 = 1.0;
  var s = Math.abs(value);
  return b1 + ((s-a1)*(b2-b1)) / (a2-a1);
};

  // Truncate a number to a certain number of decimal points:
NeuronNetworkRenderer.prototype.truncate = function(n, decimals) {
  return parseFloat(n).toFixed(Math.max(Math.min(decimals, 20), 0));
};

  // Calculate where on the inter-neuron lines to position text:
 NeuronNetworkRenderer.prototype.calculateLineTextPosition = function(originVector, destinationVector, index) {
  // Index is just used to switch between even and odd for spacing of weights
  var originToDestination = originVector.subtract(destinationVector);
  var scaled = originToDestination.scale((index % 5) * 0.03 + 0.1);
  var result = destinationVector.add(scaled);
  return result;
};

// Update the screen. 
// Show current output values if 'shownOutputs' is true
// Pass 'weights' or 'inputs' as 'lineQuantity' to select the quantity shown on the inter-neuron lines
NeuronNetworkRenderer.prototype.update = function() {

  if (this.config.lines.render && !_.contains(["weights", "inputs"], this.config.lines.quantity)) throw ("Invalid lineQuantity value: '" + this.config.lines.quantity + "'. Value must be 'weights' or 'inputs'");
  var linesAreInputs = (this.config.lines.quantity === "inputs");
  var linesAreWeights = (this.config.lines.quantity === "weights");

  var layers = this.network.getLayers();

  // Find the highest and lowest magnitude weights/inputs, for choosing inter-neuron line width:
  var highestQuantityMagnitude = -Infinity;
  var lowestQuantityMagnitude = Infinity;
  for (var layerIndex=0; layerIndex < layers.length; layerIndex++) {
    var layer = layers[layerIndex];
    for (var neuronIndex=0; neuronIndex < layer.length; neuronIndex++) {
      var neuron = layer[neuronIndex];
      for (var inputIndex=1; inputIndex < neuron.weights.length; inputIndex++) {
        var absQuantity = 0;
        if (linesAreInputs) {
          absQuantity = Math.abs(neuron.inputs[inputIndex].getOutput());
        } else if (linesAreWeights) {
          absQuantity = Math.abs(neuron.weights[inputIndex]);
        }
        if (absQuantity > highestQuantityMagnitude) highestQuantityMagnitude = absQuantity;
        if (absQuantity < lowestQuantityMagnitude) lowestQuantityMagnitude = absQuantity;
      }
    }
  }

  // Clear the screen:
  this.context.fillStyle = this.config.backgroundColor;
  this.context.fillRect(0, 0, this.context.canvas.width, this.context.canvas.height);
  
  for (var layerIndex=0; layerIndex < layers.length; layerIndex++) {
    var layer = layers[layerIndex];
    for (var neuronIndex=0; neuronIndex < layer.length; neuronIndex++) {
      var neuron = layer[neuronIndex];
      var currentCoordinatesVector = this.calculateNeuronPosition(layerIndex, neuronIndex);

      // Draw lines and labels between neurons:
      if ((layerIndex > 0) && this.config.lines.render) {
        for (var inputIndex=1; inputIndex < neuron.weights.length; inputIndex++) {
          var inputCoordinatesVector = this.calculateNeuronPosition(layerIndex-1, inputIndex-1);
          var weight = neuron.weights[inputIndex];
          var output = neuron.inputs[inputIndex].getOutput();

          // Select the width and color of the lines between neurons:
          var labelValue = linesAreInputs ? output : (linesAreWeights ? weight : "unknown");
          this.context.lineWidth = 3*Math.pow(this.normalizeQuantityForColoring(labelValue, lowestQuantityMagnitude, highestQuantityMagnitude), 4) + 0.1;
          this.context.strokeStyle = (labelValue >= 0) ? this.config.lines.positiveColor : this.config.lines.negativeColor;
          
          // Draw lines between neurons:
          this.context.beginPath();
          this.context.moveTo(currentCoordinatesVector.x, currentCoordinatesVector.y);
          this.context.lineTo(inputCoordinatesVector.x, inputCoordinatesVector.y);
          this.context.closePath();
          this.context.stroke();

          // Draw text on lines (after drawing lines, so that we write over them):
          if (this.config.lines.renderLabels) {
            this.context.fillStyle = this.config.lines.fontColor;
            this.context.font = this.config.lines.font;
            var weightPos = this.calculateLineTextPosition(inputCoordinatesVector, currentCoordinatesVector, inputIndex);
            var labelTextValue = this.truncate(labelValue, this.config.lines.labelDecimals);
            var labelText = "";
            if (labelTextValue >= 0) {
              labelText = " " + labelTextValue;
            } else {
              labelText = "" + labelTextValue;
            }
            this.context.fillText(labelText, weightPos.x-10, weightPos.y);
          }
        }
      }

      if (this.config.neurons.render) {
        // Draw the neuron as a circle, or, if provided, as an image:
        if ((this.config.neurons.image == null) || (this.config.neurons.image == undefined) || (!this.config.neurons.useImage)) {
          this.context.fillStyle = this.config.neurons.neuronColor;
          this.context.beginPath();
          this.context.arc(currentCoordinatesVector.x, currentCoordinatesVector.y, this.config.neurons.neuronRadius, 0, Math.PI*2, true); 
          this.context.closePath();
          this.context.fill();
        } else {
          var drawWidth = this.config.neurons.image.width*this.config.neurons.imageScale;
          var drawHeight = this.config.neurons.image.height*this.config.neurons.imageScale;
          this.context.drawImage(this.config.neurons.image, currentCoordinatesVector.x-(drawWidth/2), currentCoordinatesVector.y-(drawHeight/2), drawWidth, drawHeight);
        }

        // Draw text next to neurons:
        var output = neuron.getOutput();
        this.context.fillStyle = this.config.neurons.fontColor;
        this.context.font = this.config.neurons.font;
        var showOutput = this.config.neurons.renderOutputs && (output != unsetValue);
        var showBias = this.config.neurons.renderBiases && (neuron.weights.length > 0);
        if (showOutput && showBias) {
          this.context.fillText(this.config.neurons.textBiases + this.truncate(neuron.weights[0], this.config.neurons.decimalsBiases), currentCoordinatesVector.x+this.config.neurons.horizontalTextOffset, currentCoordinatesVector.y-3);
          this.context.fillText(this.config.neurons.textOutputs + this.truncate(output, this.config.neurons.decimalsOutputs), currentCoordinatesVector.x+this.config.neurons.horizontalTextOffset, currentCoordinatesVector.y+10);
        } else if (showBias) {
          this.context.fillText(this.config.neurons.textBiases + this.truncate(neuron.weights[0], this.config.neurons.decimalsBiases), currentCoordinatesVector.x+this.config.neurons.horizontalTextOffset, currentCoordinatesVector.y+3);
        } else if (showOutput) {
          this.context.fillText(this.config.neurons.textOutputs + this.truncate(output, this.config.neurons.decimalsOutputs), currentCoordinatesVector.x+this.config.neurons.horizontalTextOffset, currentCoordinatesVector.y+3);
        }
      }

    }
  }
};

var devRun = function() {

  var trainingSets = {
    xor: {
      inputs:  [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0],    [1],    [1],    [0]   ],
      inputWidth: 2,
      outputWidth: 1,
      hiddenLayers: 1,
      hiddenWidth: 2,
      learningRate: 0.1
    },
    and: {
      inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0], [0], [0], [1]],
      inputWidth: 2,
      outputWidth: 1,
      hiddenLayers: 0,
      hiddenWidth: 0,
      learningRate: 0.1
    },
    or: {
      inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0], [1], [1], [1]],
      inputWidth: 2,
      outputWidth: 1,
      hiddenLayers: 0,
      hiddenWidth: 0,
      learningRate: 0.1
    },
    nand: {
      inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[1], [1], [1], [0]],
      inputWidth: 2,
      outputWidth: 1,
      hiddenLayers: 0,
      hiddenWidth: 0,
      learningRate: 0.1
    },
    swap: {
      inputs: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
      outputs: [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
      inputWidth: 3,
      outputWidth: 3,
      hiddenLayers: 0,
      hiddenWidth: 0,
      learningRate: 0.1
    },
    test7: {
      inputs: [[0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1, 1]],
      outputs: [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
      inputWidth: 7,
      outputWidth: 4,
      hiddenLayers: 0,
      hiddenWidth: 0,
      learningRate: 0.1
    },
    mirrorThroughBottleneck2: {
      inputs:  [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      inputWidth: 2,
      outputWidth: 2,
      hiddenLayers: 1,
      hiddenWidth: 1,
      learningRate: 0.1
    },
    mirrorThroughBottleneck4: {
      inputs:  [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]],
      outputs: [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]],
      inputWidth: 4,
      outputWidth: 4,
      hiddenLayers: 1,
      hiddenWidth: 3,
      learningRate: 0.2
    }
  };

  var trainingSet = trainingSets.xor;

  var network = new NeuronNetwork(trainingSet.inputWidth, trainingSet.outputWidth, trainingSet.hiddenWidth, trainingSet.hiddenLayers);

  var learningRate = trainingSet.learningRate;

  var trainer = new BackPropagationTrainer(network, trainingSet.inputs, trainingSet.outputs);

  var canvas = document.getElementById("neuron-canvas");
  canvas.width = Math.max(trainingSet.inputWidth, trainingSet.outputWidth, trainingSet.hiddenWidth) * 150;
  canvas.height = (2 + trainingSet.hiddenLayers) * 200;
  var context = canvas.getContext("2d");

  var rendererConfig = {
    lines: {
      render: true,
      renderLabels: true,
      positiveColor: "#169e16",
      negativeColor: "#a60f0f",
      quantity: "weights",
      font: "8px Consolas",
      fontColor: "#000000",
      labelDecimals: 2
    },
    neurons: {
      render: true,
      renderBiases: true,
      renderOutputs: true,
      textBiases: "wBias: ",
      textOutputs: "out: ",
      decimalsBiases: 3,
      decimalsOutputs: 3,
      font: "10px Consolas",
      fontColor: "#000d3d",
      neuronRadius: 5,
      neuronColor: "#5e1796",
      useImage: true,
      image: document.getElementById("kawaiineuron"),
      imageScale: 1.0,
      horizontalTextOffset: 15
    },
    backgroundColor: "#ffffff",
    horizontalPadding: 50,
    verticalPadding: 0
  };

  var renderer = new NeuronNetworkRenderer(context, network, rendererConfig);

  var totalTrainings = 0;

  window.setInterval(function() {
    var trainingsPerCall = 100;
    trainer.train(trainingsPerCall, learningRate);
    totalTrainings += trainingsPerCall;
    $("#training-iterations").text("Iterations: " + totalTrainings);
  }, 30);

  window.setInterval(function() {
    var trunk = function(val) {
      return parseFloat(val).toFixed(5);
    };
    log("Results: ");
    var totalError = 0;
    /*_.each(trainingSet.inputs, function(input) {
      var result = network.evaluate(input);
      log("Expected: " + input + " Actual: " + result);
      var currentError = 0;
      for (var i=0; i < input.length; i++) {
        currentError += Math.abs(input[i] - result[i]);
      }
      totalError += currentError;
    });*/
    for (var i=0; i < trainingSet.inputs.length; i++) {
      var trainingInput = trainingSet.inputs[i];
      var trainingOutput = trainingSet.outputs[i];
      var actualOutput = network.evaluate(trainingInput);
      log("Input: " + trainingInput + " | Output: " + _.map(actualOutput, trunk));
      for (var j=0; j < trainingOutput.length; j++) {
        totalError += Math.abs(trainingOutput[j] - actualOutput[j]);
      }
    }
    log("Total error: " + totalError);
    var totalErrorString = parseFloat(totalError).toFixed(5) + "";
    var percentErrorString = parseFloat(100 * totalError / (trainingSet.outputs.length * trainingSet.outputs[0].length)).toFixed(5) + "%";
    log("Percent error: " + percentErrorString);
    $("#percent-error").text("Percent error: " + percentErrorString);
    $("#total-error").text("Total error: " + totalErrorString);
  }, 100);

  /*log("Results: ");
  //trainer.train(1000000, 1);
  _.each(trainingSet.inputs, function(input) {
    var result = network.evaluate(input);
    log("Expected: " + input + " Actual: " + result);
  });*/
  var targetFps = 55;

  var ticks = 0;
  var start = new Date();
  window.setInterval(function() {
    renderer.update();
    ticks++;
    context.font = "10px Consolas";
    context.fillStyle = "gray";
    context.fillText("FPS: " + parseFloat(ticks / ((new Date() - start)/1000)).toFixed(0), 2, 10);
    if (ticks > 100) {
      start = new Date();
      ticks = 0;
    }
  }, (1000/targetFps));

  angular.module("neuronApp", ["colorpicker.module"])
    .controller("NeuronController", ["$scope", function($scope) {
      $scope.rendererConfig = rendererConfig;
      $scope.lineQuantityChoices = ["weights", "inputs"];

      $scope.contrastingColor = function(color) {
        if (color[0] == "#") color = color.substring(1);
        try {
          return (luma(color) >= 165) ? '000' : 'fff';
        } catch (e) { // If user inputs something like "red" or "blue", just return white when the parsing error is thrown inside hexToRGBArray
          return '000';
        }
      };

      var luma = function(color) {
        var rgb = (typeof color === "string") ? hexToRGBArray(color) : color;
        return (0.2126 * rgb[0]) + (0.7152 * rgb[1]) + (0.0722 * rgb[2]); // SMPTE C, Rec. 709 weightings
      };

      var hexToRGBArray = function(color) {
        if (color.length === 3)
            color = color.charAt(0) + color.charAt(0) + color.charAt(1) + color.charAt(1) + color.charAt(2) + color.charAt(2);
        else if (color.length !== 6)
            throw("Invalid hex color: " + color);
        var rgb = [];
        for (var i = 0; i <= 2; i++)
            rgb[i] = parseInt(color.substr(i * 2, 2), 16);
        return rgb;
      };

    }]);
};

devRun();

// To switch from xy to yx, switch the coordinates to be returned opposite, switch the size decision, and switch the "calculateNeuronPosition" stuff to be backwards

