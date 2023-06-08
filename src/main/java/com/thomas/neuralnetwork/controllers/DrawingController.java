package com.thomas.neuralnetwork.controllers;

import com.thomas.neuralnetwork.data.UserData;
import javafx.fxml.FXML;

public class DrawingController {

    private final int[][] pixels = new int[28][28];

    private final String IMAGES_FILENAME = "user-images.idx3-ubyte";
    private final String LABELS_FILENAME = "user-labels.idx1-ubyte";
    private final UserData userData = UserData.getInstance();

    private void setupFiles() {
        if (userData.fileNotFound(IMAGES_FILENAME)) {
            userData.setupImageFile(IMAGES_FILENAME);
        }

        if (userData.fileNotFound(LABELS_FILENAME)) {
            userData.setupLabelFile(LABELS_FILENAME);
        }
    }

    @FXML
    private void appendUserDatapoint() {
        String numberString = null;
        int number = Integer.parseInt(numberString);


        // Convert the array of pixels to a byte array
        byte[] imageData = userData.convertPixelsToByteArray(pixels);

        // Append the image to the user test images file
        userData.addToFile(IMAGES_FILENAME, imageData);

        // Append the label to the user test labels file
        userData.addToFile(LABELS_FILENAME, new byte[]{(byte) number});
    }
}
