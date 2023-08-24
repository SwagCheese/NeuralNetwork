package com.thomas.neuralnetwork.drawing;

import java.util.Arrays;

// This preprocessor still needs some work, for example it tends to oversize smaller things (a dot will turn into a giant circle), but overall does a good job.
public class Preprocessor {
    private static double[][] getSignificantPortion(double[][] digit) {
        int min = Math.max(digit.length, digit[0].length);
        int max = 0;

        for (int i = 0; i < digit.length; i++) {
            for (int j = 0; j < digit[i].length; j++) {
                if (digit[i][j] != 0) {
                    min = Math.min(min, Math.min(i, j));
                    max = Math.max(max, Math.max(i, j));
                }
            }
        }

        double[][] res = new double[max-min][];
        for (int i = min; i < max; ++i) {
            res[i-min] = Arrays.copyOfRange(digit[i], min, max);
        }

        return res;
    }

    public static double[][] normalizeDigit(double[][] digit) {
        double[][] normalized = getSignificantPortion(digit);
        normalized = scaleDown(normalized, 20, 20);

        // Create the centered 28x28 grid
        double[][] centeredDigit = new double[28][28];

        // Calculate the starting position to center the 20x20 array
        int startX = (28 - 20) / 2;
        int startY = (28 - 20) / 2;

        // Copy the normalized digit to the centered grid
        for (int i = 0; i < 20; i++) {
            System.arraycopy(normalized[i], 0, centeredDigit[startY + i], startX, 20);
        }

        return centeredDigit;
    }

    public static double[][] blur2DArray(double[][] inputArray, int radius) {
        --radius; // assume that the radius accounts for the middle, which we don't need

        int width = inputArray.length;
        int height = inputArray[0].length;
        double[][] blurredArray = new double[width][height];

        // Iterate over each pixel in the array
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                // Variables to calculate average of surrounding pixels
                double sum = 0.0;
                int count = 0;

                // Iterate over surrounding pixels within a neighborhood defined by the radius
                for (int k = i - radius; k <= i + radius; k++) {
                    for (int l = j - radius; l <= j + radius; l++) {
                        // Check if the surrounding pixel is within bounds
                        if (k >= 0 && k < width && l >= 0 && l < height) {
                            sum += inputArray[k][l];
                            count++;
                        }
                    }
                }

                // Calculate the average and assign it to the blurred array
                blurredArray[i][j] = sum / count;
            }
        }

        return blurredArray;
    }

    public static double[][] scaleDown(double[][] input, int outputWidth, int outputHeight) {
        int inputWidth = input[0].length;
        int inputHeight = input.length;

        double[][] output = new double[outputHeight][outputWidth];

        double xRatio = ((double) inputWidth - 1) / outputWidth;
        double yRatio = ((double) inputHeight - 1) / outputHeight;

        for (int y = 0; y < outputHeight; y++) {
            int yFloor = (int) Math.floor(y * yRatio);
            int yCeiling = (int) Math.ceil(y * yRatio);

            for (int x = 0; x < outputWidth; x++) {
                int xFloor = (int) Math.floor(x * xRatio);
                int xCeiling = (int) Math.ceil(x * xRatio);

                double xLerp = (x * xRatio) - xFloor;
                double yLerp = (y * yRatio) - yFloor;

                double topLeft = input[yFloor][xFloor];
                double topRight = input[yFloor][xCeiling];
                double bottomLeft = input[yCeiling][xFloor];
                double bottomRight = input[yCeiling][xCeiling];

                double top = topLeft + (topRight - topLeft) * xLerp;
                double bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
                double value = top + (bottom - top) * yLerp;

                output[y][x] = value;
            }
        }

        return output;
    }

}
