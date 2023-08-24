package com.thomas.neuralnetwork.math.loss;

import com.thomas.neuralnetwork.math.activation.ActivationFunction;
import com.thomas.neuralnetwork.math.activation.SigmoidActivation;
import com.thomas.neuralnetwork.math.activation.TanhActivation;

public class MeanSquaredLoss implements LossFunction {
    @Override
    public double calculate(double[] actual, double[] predicted) {
        double loss = 0;

        for (int i = 0; i < actual.length; ++i) {
            loss += Math.pow(actual[i] - predicted[i], 2);
        }

        return loss/actual.length;
    }

    @Override
    public double[] derive(ActivationFunction activationFunction, double[] actual, double[] preactivation) {
        double[] derived = new double[actual.length];

        if (activationFunction instanceof SigmoidActivation) {
            double[] sigmoidActivated = new SigmoidActivation().apply(preactivation);
            double[] sigmoidDerived = new SigmoidActivation().derive(preactivation);

            for (int i = 0; i < derived.length; ++i) {
                derived[i] = 2 * (sigmoidActivated[i] - actual[i]) * sigmoidDerived[i];
            }
        } else if (activationFunction instanceof TanhActivation) {
            double[] tanhActivated = new TanhActivation().apply(preactivation);
            double[] tanhDerived = new TanhActivation().derive(preactivation);

            for (int i = 0; i < derived.length; ++i) {
                derived[i] = 2 * (tanhActivated[i] - actual[i]) * tanhDerived[i];
            }
        } else {
            throw new UnsupportedOperationException("Not implemented yet.");
        }

        return  derived;
    }
}
