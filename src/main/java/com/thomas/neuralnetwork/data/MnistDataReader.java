package com.thomas.neuralnetwork.data;

import java.io.*;

public class MnistDataReader  {
    public static double[][] readImageData(InputStream imageData) throws IOException {
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(imageData));
        dataInputStream.readInt(); // There is a "magic number" here that we do not need
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        double[][] data = new double[numberOfItems][];

        for(int i = 0; i < numberOfItems; i++) {
            double[] image = new double[nRows*nCols];
            for (int o = 0; o < nRows*nCols; o++) {
                image[o] = ((double) dataInputStream.readUnsignedByte())/255;
            }

            data[i] = image;
        }

        dataInputStream.close();

        return data;
    }

    public static double[][] readLabelData(InputStream labelData) throws IOException {
        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(labelData));
        labelInputStream.readInt(); // There is a "magic number" here that we do not need
        int numberOfLabels = labelInputStream.readInt();

        double[][] data = new double[numberOfLabels][];

        for(int i = 0; i < numberOfLabels; i++) {
            int label = labelInputStream.readUnsignedByte();

            double[] labelArray = new double[10];
            labelArray[label] = 1;

            data[i] = labelArray;
        }

        labelInputStream.close();

        return data;
    }
}
