package com.thomas.neuralnetwork.controllers;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.stage.Stage;
import javafx.scene.Parent;

public class MainController extends Application {
    /**
     * The main entry point for all JavaFX applications.
     * The start method is called after the init method has returned,
     * and after the system is ready for the application to begin running.
     *
     * <p>
     * NOTE: This method is called on the JavaFX Application Thread.
     * </p>
     *
     * @param primaryStage the primary stage for this application, onto which
     *                     the application scene can be set.
     *                     Applications may create other stages, if needed, but they will not be
     *                     primary stages.
     * @throws Exception if something goes wrong
     */
    @Override
    public void start(Stage primaryStage) throws Exception {
        Parent root = new FXMLLoader().load(getClass().getResourceAsStream("/fxml/TrainingWindow.fxml"));
        primaryStage.setTitle("MNIST Neural Network");
        primaryStage.setScene(new Scene(root, 300, 400));
        primaryStage.show();

        // Set a handler for the close request event
        primaryStage.setOnCloseRequest(event -> {
            event.consume(); // Consume the event to prevent immediate window closure

            // Show the confirmation dialog
            Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
            alert.setTitle("Confirm Close");
            alert.setHeaderText("Are you sure you want to close the window?");
            alert.setContentText("Any unsaved training will be lost.");

            // Set the buttons for the dialog
            ButtonType buttonTypeYes = new ButtonType("Yes");
            ButtonType buttonTypeNo = new ButtonType("No");
            alert.getButtonTypes().setAll(buttonTypeYes, buttonTypeNo);

            // Wait for user response
            alert.showAndWait().ifPresent(buttonType -> {
                if (buttonType == buttonTypeYes) {
                    // Close the window
                    primaryStage.close();
                }
            });
        });
    }

    @Override
    public void stop() {
        System.exit(0);
    }
}
