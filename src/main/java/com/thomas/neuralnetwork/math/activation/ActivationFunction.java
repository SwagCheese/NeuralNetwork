package com.thomas.neuralnetwork.math.activation;

public interface ActivationFunction {
    double[] apply(double[] array);

    double[] derive(double[] array);
}
