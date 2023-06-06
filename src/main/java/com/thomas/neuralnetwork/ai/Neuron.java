package com.thomas.neuralnetwork.ai;

import java.util.Arrays;

public class Neuron {
	private final double[] connections;
	private double bias;

	/**
	 * Creates a new Neuron with the specified weights and biases
	 * 
	 * @param connections the weights to use
	 * @param bias the bias for this neuron
	 */
	public Neuron(double[] connections, double bias) {
		this.connections = connections;
		this.bias = bias;
	}

	/**
	 * Adds a specified amount to the bias
	 *
	 * @param addend the amount to add to the bias
	 */
	public void addToBias(double addend) {
		bias += addend;
	}
	
	/**
	 * Adds the corresponding double from the addend to its connection
	 *
	 * @param addend an array of doubles to add to the connections
	 */
	public void addToConnections(double[] addend) {
		for (int i = 0; i < connections.length; i++) {
			connections[i] += addend[i];
		}
	}

	/**
	 * Creates a deep copy of this Neuron
	 * 
	 * @return the new copied neuron
	 */
	public Neuron copy() {
		return new Neuron(Arrays.copyOf(connections, connections.length), bias);
	}

	/**
	 * Creates a new neuron from an array
	 * 
	 * @param array the array to use when creating the neuron
	 * @return the new neuron
	 */
	public static Neuron fromArray(double[] array) {
		Neuron temp = new Neuron(new double[array.length-1], array[array.length-1]);
		System.arraycopy(array, 1, temp.connections, 0, array.length - 1);
		return temp;
	}

	/**
	 * Creates an array that represents this neuron
	 * 
	 * @return an array that represents this neuron
	 */
	public double[] toArray() {
		double[] result = Arrays.copyOf(connections, connections.length+1);
		result[connections.length] = bias;
		return result;
	}
	
	/**
	 * Gets this neurons connections
	 * 
	 * @return this neurons connections
	 */
	public double[] getConnections() {
		return connections;
	}

	/**
	 * Gets this neuron's bias
	 * 
	 * @return this neuron's bias
	 */
	public double getBias() {
		return bias;
	}
}
