package com.thomas.neuralnetwork.controllers;

import com.thomas.neuralnetwork.ai.NeuralNetwork;
import com.thomas.neuralnetwork.data.DataPoint;
import com.thomas.neuralnetwork.data.MnistDataReader;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;

import static com.thomas.neuralnetwork.ai.NeuralNetwork.LOSS_FUNCTION;

public class TestingController {

    public Button testButton;
    public Button loadNetworkButton;
    private NeuralNetwork neuralNetwork;

    List<DataPoint> dataPoints;

    @FXML
    public Label output;

    @FXML
    private void initialize() throws IOException {
        // Initialize training data
        dataPoints = MnistDataReader.readData(
                getClass().getResourceAsStream("/data/t10k-images.idx3-ubyte"),
                getClass().getResourceAsStream("/data/t10k-labels.idx1-ubyte")
        );
    }

    @FXML
    public void loadNetwork(ActionEvent event) {
        File file = MnistDataReader.chooseFile(event);

        if (file == null) return; // No file has been selected.

        neuralNetwork = NeuralNetwork.fromFile(file);
        loadNetworkButton.setText("Network \"" + file.getName().split(".nnet")[0] + "\" loaded.");
    }

    @FXML
    public void test() {
        if (neuralNetwork == null) {
            testButton.setText("Load a network first.");
            return;
        }

        testButton.setText("Test");

        double[][] predictions = new double[dataPoints.size()][];
        double[][] actual = dataPoints.stream().map(DataPoint::outputs).toArray(double[][]::new);

        for (int a = 0; a < dataPoints.size(); a++) {
            predictions[a] = neuralNetwork.forwardPropagate(dataPoints.get(a).inputs());
        }

        double cost = LOSS_FUNCTION.calculate(actual, predictions);

        /*
        Calculate accuracy
         */

        double accuracy = 0;
        int[] numIncorrect = new int[10];

        for (int i = 0; i < predictions.length; ++i) {
            for (int j = 0; j < predictions[i].length; ++j) {
                if (actual[i][j] == 1) {
                    if (predictions[i][j] > 0.5) {
                        ++accuracy;
                    } else {
                        ++numIncorrect[j];
                    }
                }
            }
        }

        accuracy /= dataPoints.size();
        accuracy *= 100;


        output.setText("Cost: " + cost + "\n" +
        "Accuracy: " + new DecimalFormat("#.##").format(accuracy) + "%.\n" +
        "Incorrect classifications: " + Arrays.stream(numIncorrect).sum() + " (" + Arrays.toString(numIncorrect) + ")");
    }
}
