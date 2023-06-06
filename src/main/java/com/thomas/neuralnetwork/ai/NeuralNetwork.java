package com.thomas.neuralnetwork.ai;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;

import com.thomas.neuralnetwork.math.ActivationFunction;
import com.thomas.neuralnetwork.math.ActivationLossCombos;
import com.thomas.neuralnetwork.math.LossFunction;
import org.jetbrains.annotations.Nullable;

public class NeuralNetwork {
	// TODO move settings into a properties file (https://www.baeldung.com/java-properties)
	private static final double LEARNING_RATE = 0.01;
	private static final int BATCH_SIZE = 500;
	private static final ActivationFunction ACTIVATION_FUNCTION = ActivationFunction.TANH;
	private static final ActivationFunction OUTPUT_ACTIVATION_FUNCTION = ActivationFunction.SOFTMAX;
	private static final LossFunction LOSS_FUNCTION = LossFunction.CROSS_ENTROPY_ERROR;

	private final Layer[] layers;

	/**
	 * Creates a new neural network with random weights and biases
	 * 
	 * @param numInput     the number of neurons in the input layer
	 * @param numHidden    the number of hidden layers
	 * @param numPerHidden the number of neurons in every hidden layer
	 * @param numOutput    the number of neurons in the output layer
	 */
	public NeuralNetwork(int numInput, int numHidden, int numPerHidden, int numOutput) {
		layers = new Layer[numHidden + 1];
		layers[0] = new Layer(numPerHidden, numInput);
		for (int i = 1; i < numHidden; i++) {
			layers[i] = new Layer(numPerHidden, numPerHidden);
		}
		layers[layers.length - 1] = new Layer(numOutput, numPerHidden);
	}

	/**
	 * Creates a new neural network with random weights and biases
	 *
	 * @param layerSizes the number of neurons in each layer,
	 *                   where index 0 is the number of inputs,
	 *                   the indices from 1 to length-2 are the number of neurons in each hidden layer,
	 *                   and index length-1 is the number of neurons in the output layer
	 */
	public NeuralNetwork(int[] layerSizes) {
		layers = new Layer[layerSizes.length-1];
		for (int i = 1; i < layerSizes.length; i++) {
			layers[i-1] = new Layer(layerSizes[i], layerSizes[i-1]);
		}
	}

	/**
	 * Creates a new neural network using the given layers
	 * 
	 * @param layers the layers to use in the network
	 */
	public NeuralNetwork(Layer[] layers) {
		this.layers = layers;
	}

	/**
	 * Performs forward propagation on the neural network
	 * 
	 * @param inputs the inputs to use
	 * @return an array of the neural networks output
	 */
	public double[] forwardPropagate(double[] inputs) {
		for (int i = 0; i < layers.length; i++) {
			if (i < layers.length - 1) {
				inputs = ACTIVATION_FUNCTION.apply(layers[i].calculateInputs(inputs));
			} else {
				inputs = OUTPUT_ACTIVATION_FUNCTION.apply(layers[i].calculateInputs(inputs));
			}
		}
		return inputs;
	}

	/**
	 * Performs forward propagation on the neural network, stopping at the specified index.
	 *
	 * @param inputs    the inputs to use
	 * @param maxIndex  the index where forward propagation should stop
	 * @return          an array of the neural network's output values
	 */
	public double[] forwardPropagate(double[] inputs, int maxIndex) {
		for (int i = 0; i <= maxIndex; i++) {
			if (i < layers.length - 1) {
				inputs = ACTIVATION_FUNCTION.apply(layers[i].calculateInputs(inputs));
			} else {
				inputs = OUTPUT_ACTIVATION_FUNCTION.apply(layers[i].calculateInputs(inputs));
			}
		}
		return inputs;
	}

	/**
	 * Performs forward propagation on the neural network, stopping at the specified index.
	 * Does not perform the activation function on the layer at the specified index
	 *
	 * @param inputs    the inputs to use
	 * @param maxIndex  the index where forward propagation should stop
	 * @return          an array of the neural network's output values
	 */
	public double[] forwardPropagateNoActivation(double[] inputs, int maxIndex) {
		for (int i = 0; i <= maxIndex; i++) {
			if (i == maxIndex) {
				inputs = layers[i].calculateInputs(inputs);
			} else {
				if (i < layers.length - 1) {
					inputs = ACTIVATION_FUNCTION.apply(layers[i].calculateInputs(inputs));
				} else {
					inputs = OUTPUT_ACTIVATION_FUNCTION.apply(layers[i].calculateInputs(inputs));
				}
			}
		}
		return inputs;
	}

	/**
	 * Performs back propagation on the neural network
	 *
	 * @param inputs        The inputs to use
	 * @param outputs       The desired outputs
	 *
	 * @return              A 3D array containing the desired changes for the neural network's weights and biases
	 * 						The dimensions of the array are as follows: [layer][neuron][deltaWeights, deltaBias]
	 */

	// TODO look into Adam optimizer for the learning rate
	public double[][][] backPropagate(double[] inputs, double[] outputs) {
		/*
		 Initialize an array to hold δL/δz for each layer
		 with L being the loss/error/cost function (same thing),
		 and z being the output of the neuron before activation
		 */
		double[][] errorGradients = new double[layers.length][];

		// Initialize the last layer's δL/δz
		errorGradients[layers.length-1] = ActivationLossCombos.TANH_MSE.derivativeLossWRTPreActivation(outputs, forwardPropagateNoActivation(inputs, layers.length-1));

		// Iterate through the layers, calculating δL/δz for each one
		for (int l = layers.length-2; l >= 0; --l) {
			/*
			 δL/δa for each the previous layer is equal to ∑((δL/δz)*w) for each weight in the current layer,
			 with w being the weight connected to the neuron we are finding the error for
			 */
			double[] layerErrorSums = new double[layers[l].getNeurons().length];

			for (int j = 0; j < errorGradients[l+1].length; ++j) {
				for (int k = 0; k < layerErrorSums.length; ++k) {
					// add (δL/δz)*w to the sum
					layerErrorSums[k] += errorGradients[l+1][j] * layers[l+1].getNeurons()[j].getConnections()[k];
				}
			}


			// calculate δa/δz
			double[] layerDerivatives = ACTIVATION_FUNCTION.derive(forwardPropagateNoActivation(inputs, l));

			/*
			 δL/δz is equal to δL/δa * δa/δz
			 To achieve this we simply set index [l][j] of the gradient array to layerErrorSums[j] (δL/δa),
			 multiplied by layerDerivatives[j] (δa/δz)
			 */
			errorGradients[l] = new double[layerDerivatives.length];

			for (int j = 0; j < errorGradients[l].length; j++) {
				errorGradients[l][j] = layerErrorSums[j] * layerDerivatives[j];
			}
		}

		double[][][] deltaLayers = new double[layers.length][][];

		for (int l = 0; l < layers.length; ++l) {
			Neuron[] neurons = layers[l].getNeurons();
			double[][] deltaNeurons = new double[neurons.length][];

			for (int j = 0; j < neurons.length; ++j) {
				double[] deltaWeightsBiases = new double[neurons[j].getConnections().length + 1];

				double[] prevLayerOutput = forwardPropagate(inputs, l-1);
				for (int i = 0; i < neurons[j].getConnections().length; i++) {
					deltaWeightsBiases[i] = -LEARNING_RATE*prevLayerOutput[i]*errorGradients[l][j];
				}

				deltaWeightsBiases[deltaWeightsBiases.length-1] = -LEARNING_RATE*errorGradients[l][j];

				deltaNeurons[j] = deltaWeightsBiases;
			}

			deltaLayers[l] = deltaNeurons;
		}

		return deltaLayers;
	}

	private double[][][][] batchBackPropagate(double[][] inputs, double[][] outputs, int batchStart, int batchEnd, @Nullable ExecutorService pool) {
		int numElements = batchEnd-batchStart;

		List<double[][][]> desiredChanges = new ArrayList<>(numElements);
		CountDownLatch latch = new CountDownLatch(numElements);

		for (int i = batchStart; i < batchEnd; i++) {
			if (pool == null) {
				desiredChanges.add(backPropagate(inputs[i], outputs[i]));
			} else {
				int finalI = i;
				pool.submit(() -> {
					desiredChanges.add(backPropagate(inputs[finalI], outputs[finalI]));
					latch.countDown();
				});
			}
		}

		if (pool != null) {
			try {
				latch.await();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

		return desiredChanges.toArray(new double[0][0][0][0]);
	}

	/**
	 * Trains the neural network using given data and runs until the cost increases
	 *
	 * @param inputs        an array of inputs to train with
	 * @param outputs       an array of outputs corresponding to the inputs
	 * @param epochs        the number of times to run the backpropagation algorithm on the dataset
	 *                      (set to 0 to run indefinitely)
	 * @param noiseFreq     the probability that a number will be randomly altered
	 * @param noiseStrength a multiplier how much a randomly selected number will be randomly altered
	 *
	 * @return the network with the lowest cost across every epoch
	 */
	public NeuralNetwork fit(double[][] inputs, double[][] outputs, int epochs, double noiseFreq, double noiseStrength) {
		System.out.println("Starting training!");
		NeuralNetwork bestNetwork = copy();

		int epoch = 1;
		ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(8);

		while (epochs == 0 || epoch <= epochs) {
			System.out.println("------------------------------------------------------------");
			System.out.println("Starting epoch " + epoch + "!");

			double[][] predictions = new double[outputs.length][];

			for (int a = 0; a < outputs.length; a++) {
				predictions[a] = forwardPropagate(inputs[a]);
			}

			double costBefore = LOSS_FUNCTION.calculate(outputs, predictions);


			// Perform backpropagation and weight updates in batches to reduce memory usage and improve speed
			for (int batchNum = 0; batchNum < Math.ceilDiv(inputs.length, BATCH_SIZE); batchNum++) {
				int batchStart = batchNum*BATCH_SIZE;
				int batchEnd = Math.min((batchNum+1)*BATCH_SIZE, inputs.length);

				System.out.println("Starting backpropagation batch #" + (batchNum+1) + " at index " + batchStart + " and ending at index " + batchEnd + ".");

				double[][][][] desiredChanges = batchBackPropagate(inputs, outputs, batchStart, batchEnd, pool);
				double[][][] averageDesiredChanges = desiredChanges[0];

				for (int o = 1; o < desiredChanges.length; o++) {
					for (int x = 0; x < desiredChanges[o].length; x++) {
						for (int y = 0; y < desiredChanges[o][x].length; y++) {
							for (int z = 0; z < desiredChanges[o][x][y].length; z++) {
								averageDesiredChanges[x][y][z] += desiredChanges[o][x][y][z];
							}
						}
					}
				}

				for (int x = 0; x < averageDesiredChanges.length; ++x) {
					for (int y = 0; y < averageDesiredChanges[x].length; ++y) {
						for (int z = 0; z < averageDesiredChanges[x][y].length; ++z) {
							averageDesiredChanges[x][y][z] /= BATCH_SIZE;
							if (Math.random() < noiseFreq) {
								averageDesiredChanges[x][y][z] += (Math.random()*2 - 1)*noiseStrength;
							}
						}
					}
				}

				// Update weights and biases based on desired changes
				for (int x = 0; x < averageDesiredChanges.length; x++) {
					for (int y = 0; y < averageDesiredChanges[x].length; y++) {
						layers[x].getNeurons()[y].addToConnections(
								Arrays.copyOfRange(averageDesiredChanges[x][y], 0, averageDesiredChanges[x][y].length - 1));
						layers[x].getNeurons()[y]
								.addToBias(averageDesiredChanges[x][y][averageDesiredChanges[x][y].length - 1]);
					}
				}
			}


			predictions = new double[outputs.length][];

			for (int a = 0; a < outputs.length; a++) {
				predictions[a] = forwardPropagate(inputs[a]);
			}

			double costAfter = LOSS_FUNCTION.calculate(outputs, predictions);

				/*
				Calculate change in cost
				 */

			String costChange;
			if (costBefore == costAfter) {
				costChange = "stayed the same";
			} else {
				double costDifference = Math.abs(costBefore - costAfter);
				if (costBefore > costAfter) {
					costChange = "decreased by " + costDifference;
					bestNetwork = copy();
				} else {
					costChange = "increased by " + costDifference;
				}
			}

				/*
				Calculate accuracy
				 */

			double accuracy = 0;

			for (int i = 0; i < predictions.length; ++i) {
				int maxIndex = 0;
				double maxValue = Double.NEGATIVE_INFINITY;

				for (int j = 0; j < predictions[i].length; ++j) {
					if (predictions[i][j] > maxValue) {
						maxValue = predictions[i][j];
						maxIndex = j;
					}
				}

				if (outputs[i][maxIndex] == 1) ++accuracy;
			}

			accuracy /= outputs.length;
			accuracy *= 100;

				/*
				Calculate certainty
				 */

			double certainty = 0;

			for (int i = 0; i < outputs.length; ++i) {
				for (int j = 0; j < outputs[i].length; ++j) {
					if (outputs[i][j] == 1) {
						certainty += predictions[i][j];
					}
				}
			}

			certainty /= outputs.length;
			certainty *= 100;

				/*
				Print information about batch
				 */

			System.out.println("Cost " + costChange + "; new cost: " + costAfter + ".");
			System.out.println("Accuracy after changes: " + new DecimalFormat("#.##").format(accuracy) + "%.");
			System.out.println("Certainty after changes: " + new DecimalFormat("#.##").format(certainty) + "%.");
			System.out.println("Epoch " + epoch++ + " complete.");
		}

		pool.close();

		return bestNetwork;
	}

	/**
	 * Creates a deep copy of the neural network.
	 *
	 * @return a copy of the neural network
	 */
	public NeuralNetwork copy() {
		Layer[] copiedLayers = new Layer[layers.length];
		for (int i = 0; i < layers.length; i++) {
			copiedLayers[i] = layers[i].copy();
		}
		return new NeuralNetwork(copiedLayers);
	}

	/**
	 * Converts the neural network to a 3D array representation.
	 *
	 * @return a 3D array representation of the neural network
	 */
	public double[][][] toArray() {
		double[][][] result = new double[layers.length][][];
		for (int i = 0; i < layers.length; i++) {
			result[i] = layers[i].toArray();
		}
		return result;
	}

	/**
	 * Creates a neural network from a 3D array representation.
	 *
	 * @param array the 3D array representation of the neural network
	 * @return a new neural network created from the array
	 */
	public static NeuralNetwork fromArray(double[][][] array) {
		Layer[] layers = new Layer[array.length];
		for (int i = 0; i < array.length; i++) {
			layers[i] = Layer.fromArray(array[i]);
		}
		return new NeuralNetwork(layers);
	}

	@Override
	public String toString() {
		return Arrays.deepToString(toArray());
	}
}
