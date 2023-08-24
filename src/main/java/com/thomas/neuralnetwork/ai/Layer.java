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
				connections[o] = Math.random()*2 - 1;
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
	 * z = ∑(a*w) + b
	 * unactivated output = sum of (previous layer output neuron activated * weight connected to said neuron) + bias
	 * 
	 * @param inputs an array of inputs used to calculate the pass
	 *
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
	 * Gets this layers neurons
	 * 
	 * @return an array of the neurons in this layer
	 */
	public Neuron[] getNeurons() {
		return neurons;
	}
}
