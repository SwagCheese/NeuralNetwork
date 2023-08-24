package com.thomas.neuralnetwork.math.activation;

public class SigmoidActivation implements ActivationFunction {
    @Override
    public double[] apply(double[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = 1 / (1 + Math.exp(-array[i]));
        }
        return result;
    }

    @Override
    public double[] derive(double[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            double sigmoid = 1 / (1 + Math.exp(-array[i]));
            result[i] = sigmoid * (1 - sigmoid);
        }
        return result;
    }
}
