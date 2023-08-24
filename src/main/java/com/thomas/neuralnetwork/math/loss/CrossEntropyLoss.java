package com.thomas.neuralnetwork.math.loss;

import com.thomas.neuralnetwork.math.activation.ActivationFunction;
import com.thomas.neuralnetwork.math.activation.SoftmaxActivation;

public class CrossEntropyLoss implements LossFunction {
    @Override
    public double calculate(double[] actual, double[] predicted) {
        double epsilon = 1e-10;
        double loss = 0;

        for (int i = 0; i < actual.length; ++i) {
            loss += actual[i] * Math.log(predicted[i] + epsilon);
        }

        return -loss;
    }

    @Override
    public double[] derive(ActivationFunction activationFunction, double[] actual, double[] preactivation) {
        double[] derived = new double[actual.length];

        if (activationFunction instanceof SoftmaxActivation) {
            double[] activated = new SoftmaxActivation().apply(preactivation);

            for (int i = 0; i < derived.length; ++i) {
                derived[i] = activated[i] - actual[i];
            }
        } else {
            throw new UnsupportedOperationException("Not implemented yet.");
        }

        return derived;
    }
}
