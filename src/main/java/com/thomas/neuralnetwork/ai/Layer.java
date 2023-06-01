package com.thomas.neuralnetwork.ai;

public class Layer {
	private final Neuron[] neurons;

	/**
	 * Creates a new Layer with random weights and biases
	 * 
	 * @param numNeurons the number of neurons in this layer
	 * @param numPreviousNeurons the number of neurons in the previous layer (used to calculate how many connections each neuron should have)
	 */
	public Layer(int numNeurons, int numPreviousNeurons) {
		neurons = new Neuron[numNeurons];
		for (int i = 0; i < numNeurons; i++) {
			double[] connections = new double[numPreviousNeurons];
			for (int o = 0; o < numPreviousNeurons; o++) {
				connections[o] = Math.random()*20 - 10;
			}
			neurons[i] = new Neuron(connections, Math.random()*2 - 1);
		}
	}

	/**
	 * Creates a new Layer using the given neurons
	 * 
	 * @param neurons an array of neurons for this Layer to use
	 */
	public Layer(Neuron[] neurons) {
		this.neurons = neurons;
	}

	/**
	 * Adds the index of the input to the corresponding neuron's bias
	 * 
	 * @param inputs an array of inputs used to modify the biases
	 * @return the updated layer
	 */
	public Layer addToBiases(double[] inputs) {
		for (int i = 0; i < inputs.length; i++) {
			neurons[i].addToBias(inputs[i]);
		}
		return this;
	}

	/**
	 * Adds the index of the input to the corresponding neuron's connection
	 * 
	 * @param inputs an array of inputs used to modify the weights of the layer's connections
	 * @return the updated layer
	 */
	public Layer addToConnections(double[] inputs) {
		for (int i = 0; i < neurons.length; i++) {
			neurons[i].addToConnection(inputs[i], i);
		}
		return this;
	}

	/**
	 * Subtracts the index of the reducer from the corresponding neuron's bias
	 * 
	 * @param reducer an array of inputs used to modify the biases
	 * @return the updated layer
	 */
	public Layer subtractFromBiases(double[] reducer) {
		for (int i = 0; i < neurons.length; i++) {
			neurons[i].subtractFromBias(reducer[i]);
		}
		return this;
	}

	/**
	 * Subtracts the input layer from this layer's connections
	 * 
	 * @param inputs the layer to subtract from this layer
	 * @return the updated layer
	 */
	public Layer subtractFromConnections(Layer inputs) {
		for (int i = 0; i < neurons.length; i++) {
			for (int o = 0; o < neurons[i].getConnections().length; o++) {
				neurons[i].addToConnection(inputs.getNeurons()[i].getConnections()[o], o);
			}
		}
		return this;
	}

	/**
	 * Multiplies the input layer with this layer's biases
	 * 
	 * @param l the layer to multiply this layers biases with
	 * @return the updated layer
	 */
	public Layer multiplyBiases(Layer l) {
		for (int i = 0; i < neurons.length; i++) {
			neurons[i].multiplyBias(l.getNeurons()[i].getBias());
		}
		return this;
	}

	/**
	 * Multiplies every bias in this layer by the input
	 * 
	 * @param d the double to multiply every bias by
	 * @return the updated layer
	 */
	public Layer multiplyBiases(double d) {
		for (Neuron neuron : neurons) {
			neuron.multiplyBias(d);
		}
		return this;
	}
	
	public Layer multiplyConnectionsByBiases(Layer l) {
		for (Neuron neuron : neurons) {
			neuron.multiplyConnections(l.biasAsArray());
		}
		  return this;
	}
	
	public Layer multiplyBiasesByConnections(Layer l) {
		  for (int i = 0; i < l.neurons.length; i++) {   
			  for (int o = 0; o < neurons.length; o++) {
				  neurons[o].multiplyBias(l.getNeurons()[i].getConnections()[o]);
			  }
		  }
		  return this;
	}

	/**
	 * Creates a deep copy of this layer
	 * 
	 * @return a new layer which is identical to this one
	 */
	public Layer copy() {
		Neuron[] n = new Neuron[neurons.length];
		for (int i = 0; i < n.length; i++) {
			n[i] = neurons[i].copy();
		}
		return new Layer(n);
	}

	/**
	 * Preforms a feed forward pass and returns the output
	 * 
	 * @param inputs an array of inputs used to calculate the pass
	 * @return the result of the feed forward pass
	 */
	public double[] calculateInputs(double[] inputs) {
		double[] result = new double[neurons.length];
		for (int i = 0; i < neurons.length; i++) {
			result[i] += neurons[i].getBias();
			for (int o = 0; o < inputs.length; o++) {
				result[i] += inputs[o] * neurons[i].getConnections()[o];
			}
		}
		return result;
	}

	/**
	 * Creates a new Layer from an array
	 * 
	 * @param array an array representing a layer
	 * @return a new layer
	 */
	public static Layer fromArray(double[][] array) {
		Neuron[] neurons = new Neuron[array.length];
		for (int i = 0; i < array.length; i++) {
			neurons[i] = Neuron.fromArray(array[i]);
		}
		return new Layer(neurons);
	}
	
	/**
	 * Creates an array from this layer
	 * 
	 * @return an array representing this layer
	 */
	public double[][] toArray() {
		double[][] result = new double[neurons.length][neurons[0].getConnections().length + 1];
		for (int i = 0; i < neurons.length; i++) {
			result[i] = neurons[i].toArray();
		}
		return result;
	}
	
	/**
	 * Creates an array of this layer's biases
	 * 
	 * @return an array of this layer's biases
	 */
	public double[] biasAsArray() {
		double[] output = new double[neurons.length];
		for (int i = 0; i < neurons.length; i++) {
			output[i] = neurons[i].getBias();
		}
		return output;
	}

	/**
	 * Gets this layers neurons
	 * 
	 * @return an array of the neurons in this layer
	 */
	public Neuron[] getNeurons() {
		return neurons;
	}
}
