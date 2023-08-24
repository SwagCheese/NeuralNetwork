package com.thomas.neuralnetwork.math.loss;

import com.thomas.neuralnetwork.math.activation.ActivationFunction;

public interface LossFunction {
    default double calculate(double[][] actual, double[][] predicted) {
        double loss = 0;

        for (int i = 0; i < actual.length; ++i) {
            loss += calculate(actual[i], predicted[i]);
        }

        return loss/actual.length;
    }

    double calculate(double[] actual, double[] predicted);

    double[] derive(ActivationFunction activationFunction, double[] actual, double[] preactivation);
}
