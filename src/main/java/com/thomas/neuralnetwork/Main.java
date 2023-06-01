package com.thomas.neuralnetwork;

import com.thomas.neuralnetwork.ai.NeuralNetwork;
import com.thomas.neuralnetwork.data.MnistDataReader;
import javafx.util.Pair;

import java.nio.file.Path;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        MnistDataReader dataReader = new MnistDataReader(Path.of("/home/bob/programming/java/NeuralNetwork/data"));

        if (!dataReader.loadMnistTestingData() || !dataReader.loadMnistTrainingData()) {
            System.err.println("An error occurred while loading data.");
            System.exit(1);
        }

        NeuralNetwork neuralNetwork = new NeuralNetwork(28*28, 2, 512, 10);

        neuralNetwork.fit(Arrays.stream(dataReader.getTrainData()).map(Pair::getValue).toArray(double[][]::new), Arrays.stream(dataReader.getTrainData()).map(Pair::getKey).toArray(double[][]::new), 1000, 0.01);
    }
}
