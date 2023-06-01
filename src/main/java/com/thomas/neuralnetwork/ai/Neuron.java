package com.thomas.neuralnetwork.ai;

import java.util.Arrays;

public class Neuron {
	private double[] connections;
	private double bias;
	private double prevOutput = 0;

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
	 * @return the updated neuron
	 */
	public Neuron addToBias(double addend) {
		bias += addend;
		return this;
	}
	
	/**
	 * Adds a specified number to the connection at an index
	 * 
	 * @param addend the amount to add to the connection
	 * @param index the index of the connection
	 * @return the updated neuron
	 */
	public Neuron addToConnection(double addend, int index) {
		connections[index] += addend;
		return this;
	}
	
	/**
	 * Adds the corresponding double from the addend to its connection
	 * 
	 * @param addend an array of doubles to add to the connections
	 * @return the updated neuron
	 */
	public Neuron addToConnections(double[] addend) {
		for (int i = 0; i < connections.length; i++) {
			connections[i] += addend[i];
		}
		return this;
	}
	
	/**
	 * Subtracts the specified amount from the bias
	 * 
	 * @param d the double to subtract from the bias
	 * @return the updated neuron
	 */
	public Neuron subtractFromBias(double d) {
		bias -= d;
		return this;
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
	 * Multiplies this neuron's bias by the multiplicand
	 * 
	 * @param multiplicand the number to multiply the bias by
	 * @return the updated neuron
	 */
	public Neuron multiplyBias(double multiplicand) {
		bias *= multiplicand;
		return this;
	}
	
	/**
	 * Multiplies every connection by the multiplicand
	 * 
	 * @param multiplicand the number to multiply the connections by
	 * @return the updated neuron
	 */
	public Neuron multiplyConnections(double multiplicand) {
		for (int i = 0; i < connections.length; i++) {
			connections[i] *= multiplicand;
		}
		return this;
	}
	
	/**
	 * Multiplies every connection by its corresponding multiplicand
	 * 
	 * @param multiplicand the array to multiply the connections by
	 * @return the updated neuron
	 */
	public Neuron multiplyConnections(double[] multiplicand) {
		for (int i = 0; i < connections.length; i++) {
			connections[i] *= multiplicand[i];
		}
		return this;
	}
	
	/**
	 * Adds the connections multiplied by the corresponding activation in the previous layer
	 * 
	 * @param previousActivations an array of the previous neurons activation
	 * @return the updated neuron
	 */
	public Neuron weightedSum(double[] previousActivations) {
		for (int i = 0; i < previousActivations.length; i++) {
			bias += connections[i]*previousActivations[i];
		}
		return this;
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
	 * Sets this neuron's connection at a specified index
	 * 
	 * @param value the new weight for the connection
	 * @param index the index of the weight to change
	 */
	public void setConnection(double value, int index) {
		connections[index] = value;
	}
	
	/**
	 * Sets this neuron's connections to the ones specified
	 * 
	 * @param connections the new weight values for this neuron's connections
	 */
	public void setConnections(double[] connections) {
		this.connections = connections;
	}
	
	/**
	 * Gets this neuron's bias
	 * 
	 * @return this neuron's bias
	 */
	public double getBias() {
		return bias;
	}
	
	/**
	 * Sets this neuron's bias to the specified value
	 * 
	 * @param value the desired value for this neuron's bias
	 */
	public void setBias(double value) {
		this.bias = value;
	}
	
	/**
	 * Gets the previous output of this neuron
	 * 
	 * @return the previous output of this neuron. Returns 0 if none.
	 */
	public double getPrevOutput() {
		return prevOutput;
	}
}
