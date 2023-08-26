package com.thomas.neuralnetwork.controllers;

import com.thomas.neuralnetwork.ai.NeuralNetwork;
import com.thomas.neuralnetwork.ai.Trainer;
import com.thomas.neuralnetwork.data.DataPoint;
import com.thomas.neuralnetwork.data.MnistDataReader;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

public class TrainingController {
    private static final String SAVE_BUTTON_TEXT = "Save To File";
    private static final String SAVE_BUTTON_SAVING_TEXT = "Saving To File...";
    private static final String SAVE_BUTTON_TRAINING_TEXT = "Pause the Training to Save.";
    private static final String SAVE_BUTTON_NULL_NETWORK_TEXT = "You Must Train a Network First.";
    private static final String SAVE_BUTTON_EXISTS_TEXT = "WARNING: File Exists. Overwrite?";
    private static final String RESET_BUTTON_CONFIRM_TEXT = "Confirm Reset?";
    private static final String RESET_BUTTON_TEXT = "Reset";
    private static final String PAUSE_BUTTON_TEXT = "Pause";
    private static final String PAUSE_BUTTON_PAUSING_TEXT = "Finishing Epoch...";
    private static final String START_BUTTON_TEXT = "Start";
    private static final String START_BUTTON_RUNNING_TEXT = "Training...";
    private static final String START_BUTTON_UNINITIALIZED_TEXT = "Load or Create a Network First.";
    private static final String CREATE_BUTTON_TEXT = "Create";
    private static final String CREATE_BUTTON_CREATED_TEXT = "Created";
    private static final String CREATE_BUTTON_ERROR_TEXT = "Bad Input";

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
    public TextField hiddenLayers;
    @FXML
    public Button createButton;
    @FXML
    private LineChart<Number, Number> lineChart;

    private LineChart.Series<Number, Number> cost;
    private LineChart.Series<Number, Number> accuracy;
    private LineChart.Series<Number, Number> certainty;

    private final Logger logger = LoggerFactory.getLogger(getClass());

    private Trainer trainer;
    private NeuralNetwork neuralNetwork;
    private CopyOnWriteArrayList<DataPoint> dataPoints;
    private volatile boolean training = false;



    @FXML
    private void initialize() throws IOException {
        // Initialize training data
        dataPoints = new CopyOnWriteArrayList<>();
        dataPoints.addAll(MnistDataReader.readData(
                getClass().getResourceAsStream("/data/train-images.idx3-ubyte"),
                getClass().getResourceAsStream("/data/train-labels.idx1-ubyte")
        ));

        // Create series for the lines and add to chart
        setupLineChart();
    }

    private void train() {
        Platform.runLater(() -> startButton.setText(START_BUTTON_RUNNING_TEXT));
        training = true;
        neuralNetwork = trainer.start(dataPoints, 0, 0, 0);

        training = false;
        Platform.runLater(() -> startButton.setText(START_BUTTON_TEXT));
    }


    @FXML
    public void startTraining() {
        if (training) return;

        if (trainer == null) {
            startButton.setText(START_BUTTON_UNINITIALIZED_TEXT);
            return;
        }

        if (!createButton.getText().equals(CREATE_BUTTON_TEXT)) createButton.setText(CREATE_BUTTON_TEXT);

        new Thread(this::train).start();
    }

    @FXML
    public void pauseTraining() {
        if (!training) return;

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
            if (trainer != null) {
                trainer.stop();
                int[] layersParsed = Arrays.stream(("784, " + hiddenLayers.getText() + ", 10").split(",")).mapToInt(s -> Integer.parseInt(s.trim())).toArray();
                trainer = new Trainer(new NeuralNetwork(layersParsed), cost, accuracy, certainty);
            }

            lineChart.getData().clear();
            setupLineChart();

            resetButton.setText(RESET_BUTTON_TEXT);
        } else {
            resetButton.setText(RESET_BUTTON_CONFIRM_TEXT);
        }
    }

    private void setupLineChart() {
        cost = new LineChart.Series<>();
        cost.setName("Cost");
        accuracy = new LineChart.Series<>();
        accuracy.setName("%Accuracy");
        certainty = new LineChart.Series<>();
        certainty.setName("%Certainty");

        lineChart.getData().add(cost);
        lineChart.getData().add(accuracy);
        lineChart.getData().add(certainty);

        // Prevent a weird bug where the upper bound increases by 10 every time the chart is reset
        ((NumberAxis) lineChart.getXAxis()).setUpperBound(100);
    }

    @FXML
    public void saveToFile() {
        if (training) {
            saveButton.setText(SAVE_BUTTON_TRAINING_TEXT);

            new Thread(() -> {
                while (training) {
                    Thread.onSpinWait();
                }

                Platform.runLater(() -> saveButton.setText(SAVE_BUTTON_TEXT));
            }).start();
        }

        if (neuralNetwork == null) {
            saveButton.setText(SAVE_BUTTON_NULL_NETWORK_TEXT);
            return;
        }

        logger.info("Preparing to save to file.");
        File trained = new File(fileName.getText() + ".nnet");

        try {
            if (saveButton.getText().equals(SAVE_BUTTON_TEXT) && !trained.createNewFile()) {
                saveButton.setText(SAVE_BUTTON_EXISTS_TEXT);
                logger.warn("Save file exists.");
                return;
            }

            if (trained.delete() && trained.createNewFile()) {
                logger.info("Successfully deleted and recreated save file.");
            }

            saveButton.setText(SAVE_BUTTON_SAVING_TEXT);

            BufferedWriter writer = new BufferedWriter(new FileWriter(trained));

            logger.info("Writing neural network to file... ");
            writer.write(neuralNetwork.toString());
            logger.info("Done!");
            writer.close();
            saveButton.setText(SAVE_BUTTON_TEXT);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @FXML
    public void createNetwork() {
        try {
            int[] layersParsed = Arrays.stream(("784, " + hiddenLayers.getText() + ", 10").split(",")).mapToInt(s -> Integer.parseInt(s.trim())).toArray();

            trainer = new Trainer(new NeuralNetwork(layersParsed), cost, accuracy, certainty);

            if (startButton.getText().equals(START_BUTTON_UNINITIALIZED_TEXT)) {
                startButton.setText(START_BUTTON_TEXT);
            }

            createButton.setText(CREATE_BUTTON_CREATED_TEXT);

            new Thread(() -> {
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

                Platform.runLater(() -> createButton.setText(CREATE_BUTTON_TEXT));
            });
        } catch (Exception e) {
            createButton.setText(CREATE_BUTTON_ERROR_TEXT);
        }
    }

    @FXML
    public void loadNetwork(ActionEvent event) {
        File file = MnistDataReader.chooseFile(event);

        if (file == null) return; // No file has been selected.

        fileName.setText(file.getName().replace(".nnet", ""));
        trainer = new Trainer(NeuralNetwork.fromFile(file), cost, accuracy, certainty);
    }


    public void toggleUseUserData(ActionEvent actionEvent) throws IOException {
        List<DataPoint> dataPointListTemp = new ArrayList<>();
        if (new File("user-images.idx3-ubyte").exists() && new File("user-labels.idx1-ubyte").exists()) {
            dataPointListTemp.addAll(MnistDataReader.readData(
                    new FileInputStream("user-images.idx3-ubyte"),
                    new FileInputStream("user-labels.idx1-ubyte")
            ));
        }

        if (((CheckBox)actionEvent.getTarget()).isSelected()) {
            dataPoints.addAll(dataPointListTemp);
            System.out.println("Added " + dataPointListTemp.size() + " Data Points.");
        } else {
            dataPoints.removeAll(dataPointListTemp);
            System.out.println("Removed " + dataPointListTemp.size() + " Data Points.");
        }
    }
}
