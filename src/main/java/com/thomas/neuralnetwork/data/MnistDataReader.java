package com.thomas.neuralnetwork.data;

import javafx.util.Pair;

import java.io.*;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class MnistDataReader  {
    private final Path dataDir;
    Pair<double[], double[]>[] trainData;
    Pair<double[], double[]>[] testData;

    public MnistDataReader(Path dataDir) {
        this.dataDir = dataDir;
    }

    public Pair<double[], double[]>[] readData(String dataFilePath, String labelFilePath) throws IOException {
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        Pair<double[], double[]>[] data = new Pair[numberOfItems];

        assert numberOfItems == numberOfLabels;

        for(int i = 0; i < numberOfItems; i++) {
            int label = labelInputStream.readUnsignedByte();

            double[] image = new double[nRows*nCols];
            double[] labelArray = new double[10];
            labelArray[label] = 1;

            for (int o = 0; o < nRows*nCols; o++) {
               image[o] = ((double) dataInputStream.readUnsignedByte())/255;
            }

            data[i] = new Pair<>(labelArray, image);
        }

        dataInputStream.close();
        labelInputStream.close();

        return data;
    }

    public boolean loadMnistTrainingData() {
        File images = dataDir.resolve("train-images.idx3-ubyte").toFile();
        File labels = dataDir.resolve("train-labels.idx1-ubyte").toFile();

        try {
            trainData = readData(images.getPath(), labels.getPath());
        } catch (IOException e) {
            return false;
        }

        return true;
    }

    public boolean loadMnistTestingData() {
        File images = dataDir.resolve("t10k-images.idx3-ubyte").toFile();
        File labels = dataDir.resolve("t10k-labels.idx1-ubyte").toFile();

        try {
            testData = readData(images.getPath(), labels.getPath());
        } catch (IOException e) {
            return false;
        }

        return true;
    }

    public Pair<double[], double[]>[] getTrainData() {
        return trainData;
    }

    public Pair<double[], double[]>[] getTestData() {
        return testData;
    }
}
