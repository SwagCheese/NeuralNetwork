package com.thomas.neuralnetwork.controllers;

import com.thomas.neuralnetwork.drawing.DrawingClassifier;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.scene.Parent;

import java.io.IOException;

public class MainController extends Application {
    @Override
    public void start(Stage primaryStage) throws IOException {
        Parent root = loadFxml("/fxml/MainWindow.fxml");
        primaryStage.setTitle("MNIST Neural Network");
        primaryStage.setScene(new Scene(root, 600, 400));
        primaryStage.show();
    }

    @Override
    public void stop() {
        System.exit(0);
    }

    public void openTestWindow() throws IOException {
        Parent root = loadFxml("/fxml/TestingWindow.fxml");

        Stage stage = new Stage();
        stage.setScene(new Scene(root, 600, 400));
        stage.show();
    }

    public void openTrainWindow() throws IOException {
        Parent root = loadFxml("/fxml/TrainingWindow.fxml");

        Stage stage = new Stage();
        stage.setScene(new Scene(root, 1200, 800));
        stage.show();
    }

    public void openDrawWindow() {
        DrawingClassifier drawingClassifier = new DrawingClassifier();

        Stage stage = new Stage();
        stage.setWidth(600);
        stage.setHeight(600);
        drawingClassifier.start(stage);
    }

    private Parent loadFxml(String resourcePath) throws IOException {
        return new FXMLLoader().load(getClass().getResourceAsStream(resourcePath));
    }
}
