package com.thomas.neuralnetwork.controllers;

import com.thomas.neuralnetwork.ai.NeuralNetwork;
import com.thomas.neuralnetwork.ai.Trainer;
import com.thomas.neuralnetwork.data.MnistDataReader;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.util.Pair;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

public class TrainingController {
    private static final String saveButtonText = "Save To File";
    private static final String saveButtonSavingText = "Save To File";
    private static final String saveButtonWarningText = "WARNING: File Exists. Overwrite?";
    private static final String resetButtonConfirmText = "Confirm Reset?";
    private static final String resetButtonText = "Reset";
    private static final String pauseButtonText = "Pause";
    private static final String pauseButtonPausingText = "Finishing Epoch...";
    private static final String startButtonText = "Start";
    private static final String startButtonRunningText = "Training...";


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
    private boolean training = false;


    @FXML
    private void initialize() {
        // Initialize training data
        MnistDataReader dataReader = new MnistDataReader(Path.of(Objects.requireNonNull(getClass().getResource("/data")).getPath()));
        dataReader.loadMnistTrainingData();
        inputs = dataReader.getTrainData().stream().map(Pair::getValue).toArray(double[][]::new);
        outputs = dataReader.getTrainData().stream().map(Pair::getKey).toArray(double[][]::new);

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
        Platform.runLater(() -> startButton.setText(startButtonRunningText));
        training = true;
        neuralNetwork = trainer.start(inputs, outputs, 0, 0, 0);

        training = false;
        Platform.runLater(() -> startButton.setText(startButtonText));
    }


    @FXML
    public void startTraining() {
        if (training) return;

        new Thread(this::train).start();
    }

    @FXML
    public void pauseTraining() {
        training = false;

        pauseButton.setText(pauseButtonPausingText);

        new Thread(() -> {
            trainer.stop();

            Platform.runLater(() -> {
                startButton.setText(startButtonText);
                pauseButton.setText(pauseButtonText);
            });
        }).start();
    }

    @FXML
    public void resetNetwork() {
        if (resetButton.getText().equals(resetButtonConfirmText)) {
            trainer.stop();
            trainer = new Trainer(new NeuralNetwork(new int[]{784, 256, 128, 10}));
            lineChart.setData(null);

            resetButton.setText(resetButtonText);
        } else {
            resetButton.setText(resetButtonConfirmText);
        }
    }

    @FXML
    public void saveToFile() {
        System.out.println("Preparing to save to file.");
        File trained = new File(fileName.getText() + ".nnet");

        try {
            if (saveButton.getText().equals(saveButtonText) && !trained.createNewFile()) {
                saveButton.setText(saveButtonWarningText);
                return;
            }

            saveButton.setText(saveButtonSavingText);

            BufferedWriter writer = new BufferedWriter(new FileWriter(trained));

            System.out.print("Writing neural network to file... ");
            writer.write(neuralNetwork.toString());
            System.out.println("Done!");
            writer.close();
            saveButton.setText(saveButtonText);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
