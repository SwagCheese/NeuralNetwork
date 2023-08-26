package com.thomas.neuralnetwork.data;

import javafx.event.ActionEvent;
import javafx.scene.Node;
import javafx.stage.FileChooser;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class MnistDataReader {
    public static File chooseFile(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        FileChooser.ExtensionFilter extensionFilter = new FileChooser.ExtensionFilter("Neural Network Files (*.nnet)", "*.nnet");

        fileChooser.getExtensionFilters().add(extensionFilter);
        fileChooser.setInitialDirectory(new File("."));

        return fileChooser.showOpenDialog(((Node) event.getSource()).getScene().getWindow());
    }
    public static List<DataPoint> readData(InputStream imageData, InputStream labelData) throws IOException {
        DataInputStream imageInputStream = new DataInputStream(new BufferedInputStream(imageData));
        imageInputStream.readInt(); // There is a "magic number" here that we do not need
        int numberOfImages = imageInputStream.readInt();
        int nRows = imageInputStream.readInt();
        int nCols = imageInputStream.readInt();

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(labelData));
        labelInputStream.readInt(); // There is a "magic number" here that we do not need
        int numberOfLabels = labelInputStream.readInt();

        List<DataPoint> data = new ArrayList<>(numberOfImages);

        assert numberOfImages == numberOfLabels;

        for(int i = 0; i < numberOfImages; i++) {
            int label = labelInputStream.readUnsignedByte();

            double[] image = new double[nRows*nCols];
            double[] labelArray = new double[10];
            labelArray[label] = 1;

            for (int o = 0; o < nRows*nCols; o++) {
                image[o] = ((double) imageInputStream.readUnsignedByte())/255;
            }

            data.add(new DataPoint(image, labelArray));
        }

        imageInputStream.close();
        labelInputStream.close();

        return data;
    }
}
