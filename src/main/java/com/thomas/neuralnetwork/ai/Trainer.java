package com.thomas.neuralnetwork.ai;

import javafx.application.Platform;
import javafx.scene.chart.XYChart;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

import static com.thomas.neuralnetwork.ai.NeuralNetwork.BATCH_SIZE;
import static com.thomas.neuralnetwork.ai.NeuralNetwork.LOSS_FUNCTION;

public class Trainer {

    private NeuralNetwork neuralNetwork;
    private boolean training;
    private volatile boolean stoppedTraining;
    private int epoch;

    private XYChart.Series<Number, Number> costSeries;
    private XYChart.Series<Number, Number> accuracySeries;
    private XYChart.Series<Number, Number> certaintySeries;

    public Trainer(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public Trainer(NeuralNetwork neuralNetwork, XYChart.Series<Number, Number> costSeries, XYChart.Series<Number, Number> accuracySeries, XYChart.Series<Number, Number> certaintySeries) {
        this.neuralNetwork = neuralNetwork;

        this.costSeries = costSeries;
        this.accuracySeries = accuracySeries;
        this.certaintySeries = certaintySeries;
    }

    private void updateChart(double epoch, double cost, double accuracy, double certainty) {
        if (costSeries == null || accuracySeries == null || certaintySeries == null) return;

        Platform.runLater(() -> {
            costSeries.getData().add(new XYChart.Data<>(epoch, cost));
            accuracySeries.getData().add(new XYChart.Data<>(epoch, accuracy));
            certaintySeries.getData().add(new XYChart.Data<>(epoch, certainty));
        });
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
    public NeuralNetwork start(double[][] inputs, double[][] outputs, int epochs, double noiseFreq, double noiseStrength) {
        System.out.println("Starting training!");
        training = true;
        NeuralNetwork bestNetwork = neuralNetwork.copy();

        ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(8);

        while ((epochs == 0 || epoch <= epochs) && training) {
            System.out.println("------------------------------------------------------------");
            System.out.println("Starting epoch " + epoch + "!");

            double[][] predictions = new double[outputs.length][];

            for (int a = 0; a < outputs.length; a++) {
                predictions[a] = neuralNetwork.forwardPropagate(inputs[a]);
            }

            double costBefore = LOSS_FUNCTION.calculate(outputs, predictions);


            // Perform backpropagation and weight updates in batches to reduce memory usage and improve speed
            for (int batchNum = 0; batchNum < Math.ceilDiv(inputs.length, BATCH_SIZE); batchNum++) {
                int batchStart = batchNum*BATCH_SIZE;
                int batchEnd = Math.min((batchNum+1)*BATCH_SIZE, inputs.length);

                System.out.println("Starting backpropagation batch #" + (batchNum+1) + " at index " + batchStart + " and ending at index " + batchEnd + ".");

                double[][][][] desiredChanges = neuralNetwork.batchBackPropagate(inputs, outputs, batchStart, batchEnd, pool);
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
                        neuralNetwork.getLayers()[x].getNeurons()[y].addToConnections(
                                Arrays.copyOfRange(averageDesiredChanges[x][y], 0, averageDesiredChanges[x][y].length - 1));
                        neuralNetwork.getLayers()[x].getNeurons()[y]
                                .addToBias(averageDesiredChanges[x][y][averageDesiredChanges[x][y].length - 1]);
                    }
                }
            }


            predictions = new double[outputs.length][];

            for (int a = 0; a < outputs.length; a++) {
                predictions[a] = neuralNetwork.forwardPropagate(inputs[a]);
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
                    bestNetwork = neuralNetwork.copy();
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
			Print information about batch and update chart
			 */

            System.out.println("Cost " + costChange + "; new cost: " + costAfter + ".");
            System.out.println("Accuracy after changes: " + new DecimalFormat("#.##").format(accuracy) + "%.");
            System.out.println("Certainty after changes: " + new DecimalFormat("#.##").format(certainty) + "%.");

            // We have epoch as an argument to create an effectively final copy
            updateChart(epoch, costAfter, accuracy, certainty);

            System.out.println("Epoch " + epoch++ + " complete.");
        }

        pool.close();

        training = false;
        stoppedTraining = true;

        return bestNetwork;
    }

    public void stop() {
        if (!training) return;

        training = false;
        stoppedTraining = false;

        while (!stoppedTraining) {
            Thread.onSpinWait();
        }
    }

    public boolean isTraining() {
        return training;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }
}
