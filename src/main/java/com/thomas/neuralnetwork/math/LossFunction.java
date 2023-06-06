package com.thomas.neuralnetwork.math;

public enum LossFunction {
    MEAN_SQUARED_ERROR {
        @Override
        public double calculate(double[] actual, double[] predicted) {
            double loss = 0;

            for (int i = 0; i < actual.length; ++i) {
                loss += Math.pow(actual[i] - predicted[i], 2);
            }

            return loss/actual.length;
        }

        @Override
        public double[] gradient(double[] actual, double[] predicted) {
            double[] grad = new double[actual.length];

            for (int i = 0; i < grad.length; ++i) {
                grad[i] = 2 * (predicted[i] - actual[i]);
            }

            return grad;
        }
    },
    MEAN_ABSOLUTE_ERROR {
        @Override
        public double calculate(double[] actual, double[] predicted) {
            double loss = 0;

            for (int i = 0; i < actual.length; ++i) {
                loss += Math.abs(actual[i] - predicted[i]);
            }

            return loss/actual.length;
        }

        @Override
        public double[] gradient(double[] actual, double[] predicted) {
            double[] grad = new double[actual.length];

            for (int i = 0; i < grad.length; ++i) {
                if (predicted[i] > actual[i]) {
                    grad[i] = 1;
                } else {
                    // the case where actual[i] == predicted[i] is undefined, so we arbitrarily decide it to be -1
                    grad[i] = -1;
                }
            }

            return grad;
        }
    },
    CROSS_ENTROPY_ERROR {
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
        public double[] gradient(double[] actual, double[] predicted) {
            double epsilon = 1e-10;
            double[] grad = new double[actual.length];

            for (int i = 0; i < grad.length; ++i) {
                grad[i] = -actual[i] / (predicted[i] + epsilon);
            }

            return grad;
        }
    };

    public double calculate(double[][] actual, double[][] predicted) {
        double loss = 0;

        for (int i = 0; i < actual.length; ++i) {
            loss += calculate(actual[i], predicted[i]);
        }

        return loss/actual.length;
    }

    public abstract double calculate(double[] actual, double[] predicted);

    public abstract double[] gradient(double[] actual, double[] predicted);
}
