package com.thomas.neuralnetwork.math.activation;

public class ReLUActivation implements ActivationFunction {
    @Override
    public double[] apply(double[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = (array[i] > 0) ? array[i] : 0;
        }
        return result;
    }

    @Override
    public double[] derive(double[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = (array[i] > 0) ? 1 : 0;
        }
        return result;
    }
}
