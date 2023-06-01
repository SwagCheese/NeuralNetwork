package com.thomas.neuralnetwork.ai;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

import com.thomas.neuralnetwork.math.Calculus;
import com.thomas.neuralnetwork.math.Function;

public class NeuralNetwork {
	private static final double LEARNING_RATE = 0.1; // TODO move into a config file
	private static final Function ACTIVATION_FUNCTION = Function.TANH; // TODO move into a config file

	private Layer[] layers;

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
				inputs = layers[i].calculateInputs(inputs);
				Calculus.applyFunction(inputs, ACTIVATION_FUNCTION);
			} else {
				inputs = layers[i].calculateInputs(inputs);
			}
		}
		return inputs;
	}

	/**
	 * Performs forward propagation on the neural network
	 * 
	 * @param inputs        the inputs to use
	 * @param maxIndex the index where forward propagation should stop
	 * @return an array of the neural networks output
	 */
	public double[] forwardPropagate(double[] inputs, int maxIndex) {
		for (int i = 0; i <= maxIndex; i++) {
			if (i < layers.length - 1) {
				inputs = layers[i].calculateInputs(inputs);
				Calculus.applyFunction(inputs, ACTIVATION_FUNCTION);
			} else {
				inputs = layers[i].calculateInputs(inputs);
			}
		}
		return inputs;
	}

	/**
	 * Performs back propagation on the neural network
	 * 
	 * @param inputs the inputs to use
	 * @param outputs the desired outputs
	 * @return an array of the neural networks output
	 */
	public double[][][] backPropagate(double[] inputs, double[] outputs, int numPerBatch) {
		double[][][] toReturn = new double[layers.length][][];

		List<Double> errors = new ArrayList<>(outputs.length);

		// initialize errors (actual output - desired output)
		double[] forwardProp = forwardPropagate(inputs);
		for (int i = 0; i < outputs.length; i++) {
			errors.add((forwardProp[i] - outputs[i]) / numPerBatch);
		}

		// loop over layers and run backpropagation algorithm
		for (int i = layers.length - 1; i >= 0; i--) {
			double[] biases = forwardPropagate(inputs, i);
			double[][] result = new double[layers[i]
					.getNeurons().length][layers[i].getNeurons()[0].getConnections().length + 1];

			Double[] errorsCopy = errors.toArray(new Double[0]);
			errors.clear();

			for (int o = 0; o < biases.length; o++) {

				// new bias += error
				result[o][result[o].length - 1] = -errorsCopy[o] * LEARNING_RATE;
				double[] prevLayerValue = forwardPropagate(inputs, i - 1);
				for (int j = 0; j < result[0].length - 1; j++) {

					// new weight += error * previous output
					result[o][j] = -errorsCopy[o] * prevLayerValue[j] * LEARNING_RATE;

					// previous layer's error = error * connection weight
					if (i != 0) {
						if (errors.size() > j) {
							errors.set(j, errors.get(j) + prevLayerValue[j] * (1 - prevLayerValue[j]) * errorsCopy[o]
									* layers[i].getNeurons()[o].getConnections()[j]);
						} else {
							errors.add(prevLayerValue[j] * (1 - prevLayerValue[j]) * errorsCopy[o]
									* layers[i].getNeurons()[o].getConnections()[j]);
						}
					}
				}
			}

			double[] prevLayerValue = forwardPropagate(inputs, i - 1);

			for (int j = 0; j < errors.size(); j++) {
				errors.set(j, errors.get(j) / numPerBatch * Calculus.tanhDerivative(prevLayerValue[j]));
			}

			toReturn[layers.length - i - 1] = result;
		}

		return toReturn;
	}

	/**
	 * Trains the neural network using given data and runs until the cost increases
	 * 
	 * @param inputs      an array of inputs to train with
	 * @param outputs      an array of outputs corresponding to the inputs
	 * @param epochs the number of times to run the backpropagation algorithm on the
	 *               dataset. set to 0 to run indefinitely
	 * @param noise  the probability that a number will be randomly altered
	 */
	public void fit(double[][] inputs, double[][] outputs, int epochs, double noise) {
		int epoch = 0;
		ExecutorService pool = Executors.newFixedThreadPool(8);

		while (epochs == 0 || epoch++ <= epochs) {
			AtomicReference<Double> costBefore = new AtomicReference<>(0D);
			CountDownLatch forwardPropLatch = new CountDownLatch(outputs.length);

			for (int a = 0; a < outputs.length; a++) {
				int finalA = a;
				pool.submit(() -> {
					double[] result = forwardPropagate(inputs[finalA]);

					for (int o = 0; o < outputs[finalA].length; o++) {
						int finalO = o;
						costBefore.updateAndGet(v -> v + Math.pow(outputs[finalA][finalO] - result[finalO], 2));
					}

					forwardPropLatch.countDown();
				});
			}

			try {
				forwardPropLatch.await();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

			costBefore.updateAndGet(v -> v / outputs.length);

			double[][][][] desiredChanges = new double[inputs.length][][][];
			CountDownLatch backPropLatch = new CountDownLatch(inputs.length);

			for (int i = 1; i < inputs.length; i++) {
				int finalI = i;
				pool.submit(() -> {
					desiredChanges[finalI] = backPropagate(inputs[finalI], outputs[finalI], inputs.length);
					backPropLatch.countDown();
					System.out.println(finalI);
				});
			}

			try {
				backPropLatch.await();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}


			double[][][] finalDesiredChanges = desiredChanges[0];
			for (int o = 1; o < desiredChanges.length; o++) {
				for (int x = 0; x < finalDesiredChanges.length; x++) {
					for (int y = 0; y < finalDesiredChanges[x].length; y++) {
						for (int z = 0; z < finalDesiredChanges[x][y].length; z++) {
							finalDesiredChanges[x][y][z] += desiredChanges[o][x][y][z];
						}
					}
				}
			}

			for (int x = 0; x < finalDesiredChanges.length; x++) {
				for (int y = 0; y < finalDesiredChanges[x].length; y++) {
					for (int z = 0; z < finalDesiredChanges[x][y].length; z++) {
						finalDesiredChanges[x][y][z] /= finalDesiredChanges.length;
						if (Math.random() < noise) {
							finalDesiredChanges[x][y][z] += Math.random()*2 - 1;
						}
					}
				}
			}

			NeuralNetwork nn = copy();

			for (int x = 0; x < finalDesiredChanges.length; x++) {
				for (int y = 0; y < finalDesiredChanges[x].length; y++) {
					nn.layers[nn.layers.length - x - 1].getNeurons()[y].addToConnections(
							Arrays.copyOfRange(finalDesiredChanges[x][y], 0, finalDesiredChanges[x][y].length - 1));
					nn.layers[nn.layers.length - x - 1].getNeurons()[y]
							.addToBias(finalDesiredChanges[x][y][finalDesiredChanges[x][y].length - 1]);
				}
			}

			double costAfter = 0D;
			for (int a = 0; a < outputs.length; a++) {
				double[] result = nn.forwardPropagate(inputs[a]);
				for (int o = 0; o < outputs[a].length; o++) {
					costAfter += Math.pow(outputs[a][o] - result[o], 2);
				}
			}
			costAfter /= outputs.length;

			System.out.println(
					((costBefore.get() <= costAfter) ? ((costBefore.get() == costAfter) ? "same    " : "increase") : "decrease")
							+ "     " + costAfter);

			layers = nn.copy().layers;
			System.out.println("Epoch " + epoch + " complete.");
		}

		pool.close();
	}

	public NeuralNetwork copy() {
		Layer[] l = new Layer[layers.length];
		for (int i = 0; i < layers.length; i++) {
			l[i] = layers[i].copy();
		}
		return new NeuralNetwork(l);
	}

	public double[][][] toArray() {
		double[][][] result = new double[layers.length][][];
		for (int i = 0; i < layers.length; i++) {
			result[i] = layers[i].toArray();
		}
		return result;
	}

	public static NeuralNetwork fromArray(double[][][] array) {
		Layer[] l = new Layer[array.length];
		for (int i = 0; i < array.length; i++) {
			l[i] = Layer.fromArray(array[i]);
		}
		return new NeuralNetwork(l);
	}

	@Override
	public String toString() {
		return Arrays.deepToString(toArray());
	}
}
