package com.thomas.neuralnetwork.math.activation;

import java.util.Arrays;

public class LinearActivation implements ActivationFunction {
    @Override
    public double[] apply(double[] array) {
        return Arrays.copyOf(array, array.length);
    }

    @Override
    public double[] derive(double[] array) {
        double[] result = new double[array.length];
        Arrays.fill(result, 1);
        return result;
    }
}
