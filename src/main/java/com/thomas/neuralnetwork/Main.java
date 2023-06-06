package com.thomas.neuralnetwork;

import com.thomas.neuralnetwork.ai.NeuralNetwork;
import com.thomas.neuralnetwork.data.MnistDataReader;
import javafx.util.Pair;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        System.out.println("Loading data!");
        MnistDataReader dataReader = new MnistDataReader(Path.of("/home/bob/programming/java/NeuralNetwork/src/main/resources/data"));

        if (!dataReader.loadMnistTestingData() || !dataReader.loadMnistTrainingData()) {
            System.err.println("An error occurred while loading data.");
            System.exit(1);
        }

        System.out.println("Done loading data.");

        NeuralNetwork neuralNetwork = new NeuralNetwork(new int[]{784, 256, 128, 10});


        double[][] trainInputs = dataReader.getTrainData().stream().map(Pair::getValue).toArray(double[][]::new);
        double[][] trainOutputs = dataReader.getTrainData().stream().map(Pair::getKey).toArray(double[][]::new);

        neuralNetwork = neuralNetwork.fit(trainInputs, trainOutputs, 200, 0, 0);

        System.out.println("Preparing to save to file.");
        File trained = new File("/home/bob/programming/java/NeuralNetwork/trained.nnet");
        boolean writeToFile = true;

        try {
            if (!trained.createNewFile()) {
                System.out.println("Warning! File exists. Overwrite? [y/N]");
                Scanner s = new Scanner(System.in);
                if (!s.next().matches("[Yy]([Ee][Ss])?")) {
                    writeToFile = false;
                }
            }

            if (writeToFile) {
                BufferedWriter writer = new BufferedWriter(new FileWriter(trained));

                System.out.print("Writing neural network to file... ");
                writer.write(neuralNetwork.toString());
                System.out.println("Done!");
                writer.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


    }
}
