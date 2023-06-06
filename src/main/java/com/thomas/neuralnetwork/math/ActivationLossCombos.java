package com.thomas.neuralnetwork.math;

// This whole enum is a bit scuffed, but I can't think of a better way to do it
public enum ActivationLossCombos {
    // TODO these should be refactored b/c they all need to be activated so the preactivation array should just be an activation array
    SOFTMAX_CEE {
        @Override
        public double[] derivativeLossWRTPreActivation(double[] actual, double[] preactivation) {
            double[] derived = new double[actual.length];
            double[] activated = ActivationFunction.SOFTMAX.apply(preactivation);

            for (int i = 0; i < derived.length; ++i) {
                derived[i] = activated[i] - actual[i];
            }

            return derived;
        }
    },
    SIGMOID_MSE {
        @Override
        public double[] derivativeLossWRTPreActivation(double[] actual, double[] preactivation) {
            double[] derived = new double[actual.length];
            double[] sigmoidActivated = ActivationFunction.SIGMOID.apply(preactivation);
            double[] sigmoidDerived = ActivationFunction.SIGMOID.derive(preactivation);

            for (int i = 0; i < derived.length; ++i) {
                derived[i] = 2 * (sigmoidActivated[i] - actual[i]) * sigmoidDerived[i];
            }

            return derived;
        }
    },
    TANH_MSE {
        @Override
        public double[] derivativeLossWRTPreActivation(double[] actual, double[] preactivation) {
            double[] derived = new double[actual.length];
            double[] tanhActivated = ActivationFunction.TANH.apply(preactivation);
            double[] tanhDerived = ActivationFunction.TANH.derive(preactivation);

            for (int i = 0; i < derived.length; ++i) {
                derived[i] = 2 * (tanhActivated[i] - actual[i]) * tanhDerived[i];
            }

            return derived;
        }
    };

    public abstract double[] derivativeLossWRTPreActivation(double[] actual, double[] preactivation);
}
