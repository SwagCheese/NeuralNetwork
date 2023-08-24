package com.thomas.neuralnetwork.math.activation;

import java.util.Arrays;

public class SoftmaxActivation implements ActivationFunction {
    @Override
    public double[] apply(double[] array) {
        double[] result = new double[array.length];
        double eSum = 0;

        for (int i = 0; i < array.length; i++) {
            result[i] = Math.exp(array[i]);
            eSum += result[i];
        }

        for (int i = 0; i < array.length; i++) {
            result[i] /= eSum;
        }

        return result;
    }

    @Override
    public double[] derive(double[] array) {
        double[][] result = new double[array.length][array.length];
        double[] softmaxArray = apply(array);

        for (int i = 0; i < array.length; ++i) {
            for (int j = 0; j < array.length; ++j) {
                if (i == j) {
                    result[i][j] = softmaxArray[i] * (1 - softmaxArray[j]);
                } else {
                    result[i][j] = softmaxArray[i] * -softmaxArray[j];
                }
            }
        }

        return Arrays.stream(result).flatMapToDouble(Arrays::stream).toArray();
    }
}
