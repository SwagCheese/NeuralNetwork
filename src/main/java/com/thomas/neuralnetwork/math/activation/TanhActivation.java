package com.thomas.neuralnetwork.math.activation;

public class TanhActivation implements ActivationFunction {
    @Override
    public double[] apply(double[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = Math.tanh(array[i]);
        }
        return result;
    }

    @Override
    public double[] derive(double[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            double tanh = Math.tanh(array[i]);
            result[i] = 1 - Math.pow(tanh, 2);
        }
        return result;
    }
}
