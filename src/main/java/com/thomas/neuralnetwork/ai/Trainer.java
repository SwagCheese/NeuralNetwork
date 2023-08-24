package com.thomas.neuralnetwork.ai;

import com.thomas.neuralnetwork.data.DataPoint;
import com.thomas.neuralnetwork.math.learningRate.CyclicLearningRate;
import com.thomas.neuralnetwork.math.learningRate.LearningRate;
import javafx.application.Platform;
import javafx.scene.chart.XYChart;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.*;

import static com.thomas.neuralnetwork.ai.NeuralNetwork.LOSS_FUNCTION;

public class Trainer {
    private static final int BATCH_SIZE = 32;


    private final NeuralNetwork neuralNetwork;
    private boolean training;
    private volatile boolean stoppedTraining;
    private int epoch;
    private final ExecutorService pool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    private final Logger logger = LoggerFactory.getLogger(getClass());

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
     * @param dataPoints    a list of data-points (inputs and outputs) to train on
     * @param epochs        the number of times to run the backpropagation algorithm on the dataset
     *                      (set to 0 to run indefinitely)
     * @param noiseFreq     the probability that a number will be randomly altered
     * @param noiseStrength a multiplier how much a randomly selected number will be randomly altered
     *
     * @return the network with the lowest cost across every epoch
     */
    public NeuralNetwork start(CopyOnWriteArrayList<DataPoint> dataPoints, int epochs, double noiseFreq, double noiseStrength) {
        logger.info("Starting training!");

        training = true;

        NeuralNetwork bestNetwork = neuralNetwork.copy();
        ThreadPoolExecutor pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(4);
        int batchesPerEpoch = Math.ceilDiv(dataPoints.size(), BATCH_SIZE);
        LearningRate learningRate = new CyclicLearningRate(0.01, 0.1, 2, 1.5, 0.9, batchesPerEpoch);

        while ((epochs == 0 || epoch <= epochs) && training) {
            double[][] inputs = dataPoints.stream().map(DataPoint::inputs).toArray(double[][]::new);
            double[][] outputs = dataPoints.stream().map(DataPoint::outputs).toArray(double[][]::new);

            logger.info("");
            logger.info("Starting epoch " + epoch + "!");

            Collections.shuffle(dataPoints);

            double[][] predictions = new double[dataPoints.size()][];

            for (int a = 0; a < dataPoints.size(); a++) {
                predictions[a] = neuralNetwork.forwardPropagate(inputs[a]);
            }

            double costBefore = LOSS_FUNCTION.calculate(outputs, predictions);

            // Perform backpropagation and weight updates in batches to reduce memory usage and improve speed
            for (int batchNum = 0; batchNum < batchesPerEpoch; batchNum++) {
                double lr = learningRate.get();

                int batchStart = batchNum * BATCH_SIZE;
                int batchEnd = Math.min((batchNum + 1) * BATCH_SIZE, dataPoints.size());

                logger.debug("Starting backpropagation batch #" + (batchNum + 1) + " at index " + batchStart + " and ending at index " + batchEnd + ".");

                double[][][][] desiredChanges = batchBackPropagate(dataPoints, batchStart, batchEnd);
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

                            // Must be negative in order to traverse the loss in the "downhill" direction
                            averageDesiredChanges[x][y][z] *= -lr;

                            if (Math.random() < noiseFreq) {
                                averageDesiredChanges[x][y][z] += (Math.random() * 2 - 1) * noiseStrength;
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

            predictions = new double[dataPoints.size()][];

            for (int a = 0; a < dataPoints.size(); a++) {
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
                for (int j = 0; j < predictions[i].length; ++j) {
                    if (outputs[i][j] == 1 && predictions[i][j] > 0.5) {
                        ++accuracy;
                    }
                }
            }

            accuracy /= dataPoints.size();
            accuracy *= 100;

            /*
            Calculate certainty
             */

            double certainty = 0;

            for (int i = 0; i < dataPoints.size(); ++i) {
                for (int j = 0; j < outputs[i].length; ++j) {
                    if (outputs[i][j] == 1) {
                        certainty += predictions[i][j];
                    }
                }
            }

            certainty /= dataPoints.size();
            certainty *= 100;

			/*
			Print information about batch and update chart
			 */

            logger.info("Cost " + costChange + "; new cost: " + costAfter + ".");
            logger.info("Accuracy after changes: " + new DecimalFormat("#.##").format(accuracy) + "%.");
            logger.info("Certainty after changes: " + new DecimalFormat("#.##").format(certainty) + "%.");

            // We have epoch as an argument to create an effectively final copy
            updateChart(epoch, costAfter, accuracy, certainty);

            learningRate.update(epoch);

            logger.info("Epoch " + (epoch++) + " complete.");
        }

        pool.close();

        training = false;
        stoppedTraining = true;

        return bestNetwork;
    }

    double[][][][] batchBackPropagate(CopyOnWriteArrayList<DataPoint> dataPoints, int batchStart, int batchEnd) {
        int numElements = batchEnd-batchStart;

        double[][][][] desiredChanges = new double[numElements][][][];

        CountDownLatch latch = new CountDownLatch(numElements);

        for (int i = batchStart; i < batchEnd; i++) {
            int finalI = i;
            pool.submit(() -> {
                desiredChanges[finalI - batchStart] = neuralNetwork.backPropagate(dataPoints.get(finalI).inputs(), dataPoints.get(finalI).outputs());
                latch.countDown();
            });
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        return desiredChanges;
    }

    public void stop() {
        if (!training) return;

        training = false;
        stoppedTraining = false;

        while (!stoppedTraining) {
            Thread.onSpinWait();
        }
    }
}
