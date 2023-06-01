package com.thomas.neuralnetwork.math;

public class Calculus {
	/**
	 * Applies the specified function to the given array
	 * 
	 * @param array the array to apply the function to
	 * @param f the function to apply to the array
	 */
	public static void applyFunction(double[] array, Function f) {
		if (f == Function.LINEAR) return;

		for (int i = 0; i < array.length; i++) {
			switch (f) {
				case SIGMOID -> array[i] = sigmoid(array[i]);
				case TANH -> array[i] = tanh(array[i]);
				case LEAKY_RELU -> array[i] = leakyReLU(array[i]);
				case RELU -> array[i] = ReLU(array[i]);
			}
		}
	}

	/**
	 * Applies the specified function's derivative to the given array
	 *
	 * @param array the array to apply the function's derivative to
	 * @param f the function whose derivative to apply to the array
	 */
	public static void applyDerivativeFunction(double[] array, Function f) {
		for (int i = 0; i < array.length; i++) {
			switch (f) {
				case SIGMOID -> array[i] = sigmoidDerivative(array[i]);
				case TANH -> array[i] = tanhDerivative(array[i]);
				case LEAKY_RELU -> array[i] = leakyReLUDerivative(array[i]);
				case RELU -> array[i] = ReLUDerivative(array[i]);
				case LINEAR -> array[i] = 1;
			}
		}
	}
	
	
	
	
	/**
	 * Applies the sigmoid function to the given number
	 * 
	 * @param d The number to apply the sigmoid function to
	 * @return 1 / (1 + e^-d)
	 */
	public static double sigmoid(double d) {
		return 1 / (1 + Math.exp(-d));
	}

	/**
	 * Applies the derivative of the sigmoid function to the given number
	 * 
	 * @param d The number to apply the sigmoid function to
	 * @return sigmoid(d) * (1 - sigmoid(d))
	 */
	public static double sigmoidDerivative(double d) {
		return sigmoid(d) * (1 - sigmoid(d));
	}

	/**
	 * Applies the inverse sigmoid function to the given number
	 * 
	 * @param d The number to apply the inverse sigmoid function to
	 * @return -ln((1 / d) - 1)
	 */
	public static double inverseSigmoid(double d) {
		return -Math.log((1 / d) - 1);
	}

	/**
	 * Applies the tanh function to the given number
	 * 
	 * @param d The number to apply the tanh function to
	 * @return tanh(d)
	 */
	public static double tanh(double d) {
		return Math.tanh(d);
	}

	/**
	 * Applies the derivative of the tanh function to the given number
	 * 
	 * @param d The number to apply the derivative of the tanh function to
	 * @return 1 - tanh(d)^2
	 */
	public static double tanhDerivative(double d) {
		return 1 - Math.pow(Math.tanh(d), 2);
	}
	
	/**
	 * Applies the leaky ReLU function to the given number
	 * 
	 * @param d The number to apply the leaky ReLU function to
	 * @return (d > 0) ? d : d*0.01
	 */
	public static double leakyReLU(double d) {
		return (d > 0) ? d : d*0.01;
	}

	/**
	 * Applies the derivative of the leaky ReLU function to the given number
	 * 
	 * @param d The number to apply the derivative of the leaky ReLU function to
	 * @return (d > 0) ? 1 : 0.01
	 */
	public static double leakyReLUDerivative(double d) {
		return (d > 0) ? 1 : 0.01;
	}

	/**
	 * Applies the ReLU function to the given number
	 *
	 * @param d The number to apply the ReLU function to
	 * @return (d > 0) ? d : d*0.01
	 */
	public static double ReLU(double d) {
		return (d > 0) ? d : 0;
	}

	/**
	 * Applies the derivative of the ReLU function to the given number
	 * Note that when d == 0 the derivative in this case is chosen to be 0, although it is technically undefined
	 *
	 * @param d The number to apply the derivative of the ReLU function to
	 * @return (d > 0) ? 1 : 0.01
	 */
	public static double ReLUDerivative(double d) {
		return (d > 0) ? 1 : 0;
	}
}
