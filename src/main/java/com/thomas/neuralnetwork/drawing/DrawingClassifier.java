package com.thomas.neuralnetwork.drawing;

import com.thomas.neuralnetwork.ai.NeuralNetwork;
import com.thomas.neuralnetwork.data.UserData;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.scene.shape.StrokeLineJoin;
import javafx.stage.Stage;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Arrays;

public class DrawingClassifier {
    private NeuralNetwork neuralNetwork;

    private UserData userData;

    private double previousX;
    private double previousY;

    private static final int CANVAS_WIDTH = 250;
    private static final int CANVAS_HEIGHT = 250;
    private static final int STROKE_WIDTH = 18;

    private Canvas canvas;
    private GraphicsContext graphicsContext;
    private Label lblResult;
    private ImageView imageView;
    private TextField userLabel;

    private int[][] lastClassified;


    public void start(Stage stage) {
        neuralNetwork = NeuralNetwork.fromFile(new File("trained.nnet"));

        stage.setTitle("Drawing Classifier");

        imageView = new ImageView();
        imageView.setFitHeight(150);
        imageView.setFitWidth(150);

        canvas = new Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
        graphicsContext = canvas.getGraphicsContext2D();
        clearCanvas();

        canvas.setOnMouseDragged(this::drawOnCanvas);
        canvas.setOnMousePressed(event -> {
            previousX = event.getX();
            previousY = event.getY();
        });

        Button btnClassify = new Button("Classify");
        btnClassify.setOnAction(event -> classify());

        Button btnClear = new Button("Clear");
        btnClear.setOnAction(event -> clearCanvas());

        userLabel = new TextField();

        Button btnAddDatapoint = new Button("Add Datapoint");
        btnAddDatapoint.setOnAction(event -> addDatapoint());

        lblResult = new Label();

        HBox hbBottom = new HBox(10, canvas, imageView);
        hbBottom.setAlignment(Pos.CENTER);

        HBox buttonBox = new HBox(10, btnClassify, btnClear);
        HBox.setMargin(btnClassify, new Insets(10));
        HBox.setMargin(btnClear, new Insets(10));
        buttonBox.setAlignment(Pos.CENTER);

        HBox saveBox = new HBox(10, userLabel, btnAddDatapoint);
        HBox.setMargin(userLabel, new Insets(10));
        HBox.setMargin(btnAddDatapoint, new Insets(10));
        saveBox.setAlignment(Pos.CENTER);


        VBox root = new VBox(5, hbBottom, buttonBox, saveBox, lblResult);
        root.setAlignment(Pos.CENTER);
        BorderPane.setMargin(canvas, new Insets(10));

        graphicsContext.setLineCap(StrokeLineCap.ROUND);
        graphicsContext.setLineJoin(StrokeLineJoin.ROUND);

        stage.setScene(new Scene(root));
        stage.show();
    }

    private void addDatapoint() {
        if (userData == null) {
            userData = new UserData("user");
            userData.setupImageFile();
            userData.setupLabelFile();
        }

        if (lastClassified == null || userLabel.getText().isEmpty()) {
            return;
        }

        int label;

        try {
            label = Integer.parseInt(userLabel.getText());
        } catch (NumberFormatException e) {
            return;
        }

        userData.addToLabels(UserData.intToByteArray(label));
        userData.addToImages(UserData.pixelsToByteArray(lastClassified));
    }

    private void drawOnCanvas(MouseEvent event) {
        double mouseX = event.getX();
        double mouseY = event.getY();

        graphicsContext.setStroke(Color.BLACK);
        graphicsContext.setLineWidth(STROKE_WIDTH);
        graphicsContext.setLineCap(StrokeLineCap.ROUND);
        graphicsContext.setLineJoin(StrokeLineJoin.ROUND);

        graphicsContext.strokeLine(previousX, previousY, mouseX, mouseY);

        // Update previous coordinates
        previousX = mouseX;
        previousY = mouseY;
    }


    private void classify() {
        lastClassified = Arrays.stream(getGrayscaleValues()).map(d -> Arrays.stream(d).mapToInt(d1 -> (int) (d1 * 255)).toArray()).toArray(int[][]::new);
        double[] grayscaleValues = Arrays.stream(getGrayscaleValues()).flatMapToDouble(Arrays::stream).toArray();

        drawImage(grayscaleValues, 28, 28);

        double[] predictions = neuralNetwork.forwardPropagate(grayscaleValues);

        int finalResult = -1;

        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] > 0.5) {
                finalResult = i;
            }
        }

        if (finalResult == -1) {
            lblResult.setText("Network unsure\n" + Arrays.toString(predictions));
        } else {
            lblResult.setText("Prediction: " + finalResult + "\n" + transformToString(predictions, new DecimalFormat("#.##")));
        }
    }

    private double[][] getGrayscaleValues() {
        WritableImage image = new WritableImage(CANVAS_WIDTH, CANVAS_HEIGHT);
        canvas.snapshot(null, image);

        PixelReader pixelReader = image.getPixelReader();

        double[][] pixels = new double[CANVAS_HEIGHT][CANVAS_WIDTH];
        boolean empty = true;

        for (int i = 0; i < CANVAS_HEIGHT; ++i) {
            for (int o = 0; o < CANVAS_WIDTH; ++o) {
                pixels[i][o] = Math.abs(pixelReader.getColor(o, i).getRed()-1);

                if (pixels[i][o] != 0) empty = false;
            }
        }

        // Don't apply any fancy preprocessing to empty canvas as it causes an error
        return empty ? Preprocessor.scaleDown(pixels, 28, 28) : Preprocessor.normalizeDigit(Preprocessor.blur2DArray(pixels, 3));
    }


    private void drawImage(double[] img, int width, int height) {
        WritableImage toRender = new WritableImage(width, height);

        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                double color = img[y*height + x];
                toRender.getPixelWriter().setColor(x, y, new Color(color, color, color, 1));
            }
        }

        imageView.setImage(toRender);
    }

    public static String transformToString(double[] array, DecimalFormat decimalFormat) {
        StringBuilder sb = new StringBuilder();

        for (double value : array) {
            String formattedValue = decimalFormat.format(value);
            sb.append(formattedValue).append(", ");
        }

        // Remove the trailing comma and space
        if (sb.length() > 2) {
            sb.setLength(sb.length() - 2);
        }

        return sb.toString();
    }


    private void clearCanvas() {
        graphicsContext.setFill(Color.WHITE);
        graphicsContext.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    }
}
