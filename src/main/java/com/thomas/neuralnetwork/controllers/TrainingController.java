package com.thomas.neuralnetwork.controllers;

import com.thomas.neuralnetwork.ai.NeuralNetwork;
import com.thomas.neuralnetwork.ai.Trainer;
import com.thomas.neuralnetwork.data.MnistDataReader;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class TrainingController {
    private static final String SAVE_BUTTON_TEXT = "Save To File";
    private static final String SAVE_BUTTON_SAVING_TEXT = "Saving To File";
    private static final String SAVE_BUTTON_TRAINING_TEXT = "Pause the Training to Save.";
    private static final String SAVE_BUTTON_EXISTS_TEXT = "WARNING: File Exists. Overwrite?";
    private static final String RESET_BUTTON_CONFIRM_TEXT = "Confirm Reset?";
    private static final String RESET_BUTTON_TEXT = "Reset";
    private static final String PAUSE_BUTTON_TEXT = "Pause";
    private static final String PAUSE_BUTTON_PAUSING_TEXT = "Finishing Epoch...";
    private static final String START_BUTTON_TEXT = "Start";
    private static final String START_BUTTON_RUNNING_TEXT = "Training...";


    @FXML
    public TextField fileName;
    @FXML
    public Button saveButton;
    @FXML
    public Button resetButton;
    @FXML
    public Button startButton;
    @FXML
    public Button pauseButton;

    @FXML
    private LineChart<Number, Number> lineChart;

    private LineChart.Series<Number, Number> cost;
    private LineChart.Series<Number, Number> accuracy;
    private LineChart.Series<Number, Number> certainty;

    private Trainer trainer;
    private NeuralNetwork neuralNetwork;
    private double[][] inputs;
    private double[][] outputs;
    private volatile boolean training = false;


    @FXML
    private void initialize() throws IOException {
        // Initialize training data
        inputs = MnistDataReader.readImageData(getClass().getResourceAsStream("/data/train-images.idx3-ubyte"));
        outputs = MnistDataReader.readLabelData(getClass().getResourceAsStream("/data/train-labels.idx1-ubyte"));

        // Create series for the lines
        cost = new LineChart.Series<>();
        cost.setName("Cost");
        accuracy = new LineChart.Series<>();
        accuracy.setName("%Accuracy");
        certainty = new LineChart.Series<>();
        certainty.setName("%Certainty");

        // Add series to the chart
        lineChart.getData().add(cost);
        lineChart.getData().add(accuracy);
        lineChart.getData().add(certainty);

        // Initialize the trainer
        trainer = new Trainer(new NeuralNetwork(new int[]{784, 256, 128, 10}), cost, accuracy, certainty);

    }

    private void train() {
        Platform.runLater(() -> startButton.setText(START_BUTTON_RUNNING_TEXT));
        training = true;
        neuralNetwork = trainer.start(inputs, outputs, 0, 0, 0);

        training = false;
        Platform.runLater(() -> startButton.setText(START_BUTTON_TEXT));
    }


    @FXML
    public void startTraining() {
        if (training) return;

        new Thread(this::train).start();
    }

    @FXML
    public void pauseTraining() {
        training = false;

        pauseButton.setText(PAUSE_BUTTON_PAUSING_TEXT);

        new Thread(() -> {
            trainer.stop();

            Platform.runLater(() -> {
                startButton.setText(START_BUTTON_TEXT);
                pauseButton.setText(PAUSE_BUTTON_TEXT);
            });
        }).start();
    }

    @FXML
    public void resetNetwork() {
        if (resetButton.getText().equals(RESET_BUTTON_CONFIRM_TEXT)) {
            trainer.stop();
            trainer = new Trainer(new NeuralNetwork(new int[]{784, 256, 128, 10}));
            lineChart.setData(null);

            resetButton.setText(RESET_BUTTON_TEXT);
        } else {
            resetButton.setText(RESET_BUTTON_CONFIRM_TEXT);
        }
    }

    @FXML
    public void saveToFile() {
        if (training) {
            saveButton.setText(SAVE_BUTTON_TRAINING_TEXT);

            new Thread(() -> {
                while (training) {
                    Thread.onSpinWait();
                }

                Platform.runLater(() -> {
                    saveButton.setText(SAVE_BUTTON_TEXT);
                });
            }).start();
        }

        System.out.println("Preparing to save to file.");
        File trained = new File(fileName.getText() + ".nnet");

        try {
            if (saveButton.getText().equals(SAVE_BUTTON_TEXT) && !trained.createNewFile()) {
                saveButton.setText(SAVE_BUTTON_EXISTS_TEXT);
                return;
            }

            saveButton.setText(SAVE_BUTTON_SAVING_TEXT);

            BufferedWriter writer = new BufferedWriter(new FileWriter(trained));

            System.out.print("Writing neural network to file... ");
            writer.write(neuralNetwork.toString());
            System.out.println("Done!");
            writer.close();
            saveButton.setText(SAVE_BUTTON_TEXT);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
