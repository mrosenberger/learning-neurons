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

  /*return _.map(
    _.zip(
      _.map(outputNeurons, function(outputNeuron) { return outputNeuron.getOutput(); }), 
      expectedValues), 
    function(pair) { return calculateError(pair[1], pair[0]); }
  );*/
  for (var i=0; i < outputNeurons.length; i++) {
    //error += calculateError(expectedValues[i], outputNeurons[i].getOutput());
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
        var addAmount = (Math.random() - 0.5) * learningRate;
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

var NeuronNetworkRenderer = function(context, network, neuronImage) {
  this.context = context;
  this.network = network;
  this.neuronImage = neuronImage;
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
    35,
    0);
};

// Update the screen. 
// Show current output values if 'shownOutputs' is true
// Pass 'weights' or 'inputs' as 'lineQuantity' to select the quantity shown on the inter-neuron lines
NeuronNetworkRenderer.prototype.update = function(showOutputs, lineQuantity) {

  if (!_.contains(["weights", "inputs"], lineQuantity)) throw ("Invalid lineQuantity value: '" + lineQuantity + "'. Value must be 'weights' or 'inputs'");
  var linesAreInputs = (lineQuantity === "inputs");
  var linesAreWeights = (lineQuantity === "weights");

  var fontName = "Arial";
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

 // Normalizes using the above enclosed highest and lowest weights:
  var normalizeQuantityForColoring = function(value) {
    var a1 = lowestQuantityMagnitude;
    var a2 = highestQuantityMagnitude;
    var b1 = 0.0;
    var b2 = 1.0;
    var s = Math.abs(value);
    return b1 + ((s-a1)*(b2-b1)) / (a2-a1);
  };

  // Truncate a number to a certain number of decimal points:
  var truncate = function(n, decimals) {
    return parseFloat(n).toFixed(decimals);
  };

  // Calculate where on the inter-neuron lines to position text:
  var calculateLineTextPosition = function(originVector, destinationVector, index) {
    // Index is just used to switch between even and odd for spacing of weights
    var originToDestination = originVector.subtract(destinationVector);
    var scaled = originToDestination.scale((index % 5) * 0.03 + 0.1);
    var result = destinationVector.add(scaled);
    return result;
  };

  // Clear the screen:
  this.context.fillStyle = "#ffffff";
  this.context.fillRect(0, 0, this.context.canvas.width, this.context.canvas.height);
  
  for (var layerIndex=0; layerIndex < layers.length; layerIndex++) {
    var layer = layers[layerIndex];
    for (var neuronIndex=0; neuronIndex < layer.length; neuronIndex++) {
      var neuron = layer[neuronIndex];
      var currentCoordinatesVector = this.calculateNeuronPosition(layerIndex, neuronIndex);
      if (layerIndex > 0) {
        for (var inputIndex=1; inputIndex < neuron.weights.length; inputIndex++) {
          var inputCoordinatesVector = this.calculateNeuronPosition(layerIndex-1, inputIndex-1);
          var weight = neuron.weights[inputIndex];
          var output = neuron.inputs[inputIndex].getOutput();

          // Select the width and color of the lines between neurons:
          if (linesAreInputs) {
            this.context.lineWidth = 3*Math.pow(normalizeQuantityForColoring(output), 4) + 0.1;
            this.context.strokeStyle = (output >= 0) ? "gray" : "red";
          } else if (linesAreWeights) {
            this.context.lineWidth = 3*Math.pow(normalizeQuantityForColoring(weight), 4) + 0.1;
            //this.context.lineWidth = Math.exp(normalizeQuantityForColoring(weight)/10) + 0.1;
            this.context.strokeStyle = (weight >= 0) ? "green" : "red";
          }
          
          // Draw lines between neurons:
          this.context.beginPath();
          this.context.moveTo(currentCoordinatesVector.x, currentCoordinatesVector.y);
          this.context.lineTo(inputCoordinatesVector.x, inputCoordinatesVector.y);
          this.context.closePath();
          this.context.stroke();

          // Draw text on lines (after drawing lines, so that we write over them):
          if (linesAreInputs) {
            this.context.fillStyle = "black";
            this.context.font = "8px " + fontName;
            var weightPos = calculateLineTextPosition(inputCoordinatesVector, currentCoordinatesVector, inputIndex);
            this.context.fillText(truncate(output, 2), weightPos.x-7, weightPos.y);
          } else if (linesAreWeights) {
            this.context.fillStyle = "black";
            this.context.font = "8px " + fontName;
            var weightPos = calculateLineTextPosition(inputCoordinatesVector, currentCoordinatesVector, inputIndex);
            this.context.fillText(truncate(weight, 2), weightPos.x-7, weightPos.y);
          }
          
        }
      }

      var scale = 1.0;

      if ((this.neuronImage == null) || (this.neuronImage == undefined)) {
        this.context.fillStyle = "black";
        this.context.beginPath();
        this.context.arc(currentCoordinatesVector.x, currentCoordinatesVector.y, 5, 0, Math.PI*2, true); 
        this.context.closePath();
        this.context.fill();
      } else {
        //this.context.drawImage(this.neuronImage, currentCoordinatesVector.x-(this.neuronImage.width/2), currentCoordinatesVector.y-(this.neuronImage.height/2));
        var drawWidth = this.neuronImage.width*scale;
        var drawHeight = this.neuronImage.height*scale;
        this.context.drawImage(this.neuronImage, currentCoordinatesVector.x-(drawWidth/2), currentCoordinatesVector.y-(drawHeight/2), drawWidth, drawHeight);
      }

      // Draw text next to neurons:
      var output = neuron.getOutput();
      this.context.fillStyle = "black";
      this.context.font = "9px " + fontName;
      var showOutput = showOutputs && (output != unsetValue);
      var showBias = neuron.weights.length > 0;
      var xOffset = this.neuronImage.width*scale/2;
      if (showOutput && showBias) {
        this.context.fillText("Bias Weight: " + truncate(neuron.weights[0], 2), currentCoordinatesVector.x+xOffset, currentCoordinatesVector.y-3);
        this.context.fillText("Output: " + truncate(output, 2), currentCoordinatesVector.x+xOffset, currentCoordinatesVector.y+10);
      } else if (showBias) {
        this.context.fillText("Bias Weight: " + truncate(neuron.weights[0], 2), currentCoordinatesVector.x+xOffset, currentCoordinatesVector.y+3);
      } else if (showOutput) {
        this.context.fillText("Output: " + truncate(output, 2), currentCoordinatesVector.x+xOffset, currentCoordinatesVector.y+3);
      }
    }
  }
};

var fastUpdateDemo = function() {
  var inputWidth = 7;
  var outputWidth = 4;
  var hiddenLayers = 2;
  var hiddenWidth = 3;

  var network = new NeuronNetwork(inputWidth, outputWidth, hiddenWidth, hiddenLayers);

  var trainingSets = {
    xor: {
      inputs:  [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0],    [1],    [1],    [0]   ]
    },
    and: {
      inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0], [0], [0], [1]]
    },
    swap: {
      inputs: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
      outputs: [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    },
    test7: {
      inputs: [[0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1, 1]],
      outputs: [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    }
  };

  var trainingSet = trainingSets.test7;
  var trainer = new DumbTrainer(network, trainingSet.inputs, trainingSet.outputs);

  var canvas = document.getElementById("neuron-canvas");
  canvas.width = Math.max(inputWidth, outputWidth, hiddenWidth) * 150;
  canvas.height = (2 + hiddenLayers) * 200;
  var context = canvas.getContext("2d");

  var renderer = new NeuronNetworkRenderer(context, network, document.getElementById("kawaiineuron"));

  window.setInterval(function() {
    trainer.train(1);
  }, 30);

  window.setInterval(function() {
    renderer.update(true, "weights");
  }, 30);
};

var xorTest = function() {
  var inputWidth = 2;
  var outputWidth = 1;
  var hiddenLayers = 1;
  var hiddenWidth = 2;

  var network = new NeuronNetwork(inputWidth, outputWidth, hiddenWidth, hiddenLayers);

  var trainingSets = {
    xor: {
      inputs:  [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0],    [1],    [1],    [0]   ]
    },
    and: {
      inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0], [0], [0], [1]]
    },
    swap: {
      inputs: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
      outputs: [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    },
    test7: {
      inputs: [[0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1, 1]],
      outputs: [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    }
  };

  var trainingSet = trainingSets.xor;
  var trainer = new DumbTrainer(network, trainingSet.inputs, trainingSet.outputs);

  var canvas = document.getElementById("neuron-canvas");
  canvas.width = Math.max(inputWidth, outputWidth, hiddenWidth) * 150;
  canvas.height = (2 + hiddenLayers) * 120 + 15*Math.max(inputWidth, outputWidth, hiddenWidth)*(hiddenLayers+2);
  var context = canvas.getContext("2d");

  var renderer = new NeuronNetworkRenderer(context, network, document.getElementById("kawaiineuron"));

  window.setInterval(function() {
    trainer.train(100);
  }, 15);

  window.setInterval(function() {
    renderer.update(true, "weights");
  }, 30);
};

var devRun = function() {
  var inputWidth = 2;
  var outputWidth = 1;
  var hiddenLayers = 1;
  var hiddenWidth = 2;

  var network = new NeuronNetwork(inputWidth, outputWidth, hiddenWidth, hiddenLayers);

  var trainingSets = {
    xor: {
      inputs:  [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0],    [1],    [1],    [0]   ]
    },
    and: {
      inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      outputs: [[0], [0], [0], [1]]
    },
    swap: {
      inputs: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
      outputs: [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    },
    test7: {
      inputs: [[0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1, 1]],
      outputs: [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    }
  };

  var trainingSet = trainingSets.xor;
  var trainer = new DumbTrainer(network, trainingSet.inputs, trainingSet.outputs);

  var canvas = document.getElementById("neuron-canvas");
  canvas.width = Math.max(inputWidth, outputWidth, hiddenWidth) * 150;
  canvas.height = (2 + hiddenLayers) * 200;
  var context = canvas.getContext("2d");

  var renderer = new NeuronNetworkRenderer(context, network, document.getElementById("kawaiineuron"));

  window.setInterval(function() {
    trainer.train(1);
  }, 30);

  window.setInterval(function() {
    renderer.update(true, "weights");
  }, 30);
};

devRun();

// To switch from xy to yx, switch the coordinates to be returned opposite, switch the size decision, and switch the "calculateNeuronPosition" stuff to be backwards


// Optional display stuff:
//   drawLines, lineQuantity, positiveLineColor, negativeLineColor, drawBiases, drawOutputs, 

var rendererConfig = {
  lines: "none", // One of 'none', 'weigths', or 'inputs'
  positiveLineColor: "green",
  negativeLineColor: "red",
  drawBiases: true,
  drawOutputs: true,
  neuronImage: document.getElementById("kawaiineuron"),
  neuronImageScale: 1.0
};

var rendererConfig = {
  lines: {
    render: true,
    positiveColor: "green",
    negativeColor: "red",
    quantity: "weights",
    font: "8px Arial",
    fontColor: "black"
  },
  neurons: {
    drawBiases: true,
    drawOutputs: true,
    font: "8px Arial",
    fontColor: "black",
    image: document.getElementById("kawaiineuron"),
    imageScale: 1.0
  }
};
