package com.thomas.neuralnetwork;

import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;

import java.io.*;

public class MainController {
    @FXML
    private Canvas canvas;

    @FXML
    private Label predictionLabel;

    @FXML
    private TextField numberInput;

    private final int[][] pixels = new int[28][28];

    private final String IMAGES_FILENAME = "user-images.idx3-ubyte";
    private final String LABELS_FILENAME = "user-labels.idx1-ubyte";

    @FXML
    private void initialize() {
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        setupFiles();
    }

    @FXML
    private void drawPixel(MouseEvent event) {
        int size = 10;
        double x = event.getX();
        double y = event.getY();

        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.BLACK);
        gc.fillRect(x - (x % size), y - (y % size), size, size);

        int pixelX = (int) (x / size);
        int pixelY = (int) (y / size);
        pixels[pixelX][pixelY] = 1;
    }

    @FXML
    private void recognizeImage() {

    }

    @FXML
    private void appendUserDatapoint() {
        String numberString = numberInput.getText();
        int number = Integer.parseInt(numberString);

        // Convert the array of pixels to a byte array
        byte[] imageData = convertPixelsToByteArray(pixels);

        // Append the image to the user test images file
        addToFile(IMAGES_FILENAME, imageData);

        // Append the label to the user test labels file
        addToFile(LABELS_FILENAME, new byte[]{(byte) number});
    }

    private byte[] convertPixelsToByteArray(int[][] pixels) {
        byte[] data = new byte[784]; // 28x28 = 784
        int index = 0;

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                data[index++] = (byte) pixels[x][y];
            }
        }

        return data;
    }

    private void addToFile(String filename, byte[] data) {
        incrementFirstDimensionInFile(filename);
        try (FileOutputStream fos = new FileOutputStream(filename, true);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            bos.write(data);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void setupFiles() {
        if (!fileExists(IMAGES_FILENAME)) {
            setupImageFile();
        }

        if (!fileExists(LABELS_FILENAME)) {
            setupLabelFile();
        }
    }

    private boolean fileExists(String filename) {
        File file = new File(filename);
        return file.exists() && !file.isDirectory();
    }

    private void setupImageFile() {
        try (FileOutputStream fos = new FileOutputStream(IMAGES_FILENAME);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            /*
            Write the magic number
            The magic number is four bytes long. The first 2 bytes are always 0.

            The third byte codes the type of the data:
                0x08: unsigned byte
                0x09: signed byte
                0x0B: short (2 bytes)
                0x0C: int (4 bytes)
                0x0D: float (4 bytes)
                0x0E: double (8 bytes)

            The fourth byte codes the number of dimensions in the vector/matrix
             */
            bos.write(new byte[]{0x00, 0x00, 0x08, 0x03});

            // Write the dimensions (0 images, 28x28 pixels)
            // Dimension 1 will be updated later
            bos.write(0); // Size in dimension 1 (0)
            bos.write(new byte[]{0x00, 0x00, 0x00, 0x1C}); // Size in dimension 2 (28)
            bos.write(new byte[]{0x00, 0x00, 0x00, 0x1C}); // Size in dimension 3 (28)
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void setupLabelFile() {
        try (FileOutputStream fos = new FileOutputStream(LABELS_FILENAME);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            /*
            Write the magic number
            The magic number is four bytes long. The first 2 bytes are always 0.

            The third byte codes the type of the data:
                0x08: unsigned byte
                0x09: signed byte
                0x0B: short (2 bytes)
                0x0C: int (4 bytes)
                0x0D: float (4 bytes)
                0x0E: double (8 bytes)

            The fourth byte codes the number of dimensions in the vector/matrix
            */
            bos.write(new byte[]{0x00, 0x00, 0x08, 0x01});

            // Write the dimensions
            bos.write(0); // Number of dimensions
            bos.write(new byte[]{0x00, 0x00, 0x00, 0x00}); // Size in dimension 1 (0)
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void incrementFirstDimensionInFile(String filename) {
        try (RandomAccessFile file = new RandomAccessFile(filename, "rw")) {
            file.seek(4);
            file.writeInt(file.readInt()+1);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
