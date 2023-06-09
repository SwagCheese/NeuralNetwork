package com.thomas.neuralnetwork;

import com.thomas.neuralnetwork.controllers.MainController;
import javafx.application.Application;

public class Main {
    public static void main(String[] args) {
        // This is a necessary workaround when bundling as a jar
        // See https://stackoverflow.com/questions/59974282/how-to-bundle-the-javafx-sdk-directly-in-the-output-jar
        Application.launch(MainController.class, args);
    }
}
