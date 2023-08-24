package com.thomas.neuralnetwork.math.loss;

import com.thomas.neuralnetwork.math.activation.ActivationFunction;

public class MeanAbsoluteLoss implements LossFunction {
    @Override
    public double calculate(double[] actual, double[] predicted) {
        double loss = 0;

        for (int i = 0; i < actual.length; ++i) {
            loss += Math.abs(actual[i] - predicted[i]);
        }

        return loss/actual.length;
    }

    @Override
    public double[] derive(ActivationFunction activationFunction, double[] actual, double[] preactivation) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }
}
