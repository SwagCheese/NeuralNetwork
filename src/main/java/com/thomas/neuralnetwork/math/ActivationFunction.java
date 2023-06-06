package com.thomas.neuralnetwork.math;

import java.util.Arrays;
import java.util.stream.Stream;

public enum ActivationFunction {
    LINEAR {
        @Override
        public double[] apply(double[] array) {
            return Arrays.copyOf(array, array.length);
        }

        @Override
        public double[] derive(double[] array) {
            double[] result = new double[array.length];
            Arrays.fill(result, 1);
            return result;
        }
    },
    RELU {
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
    },
    LEAKY_RELU {
        @Override
        public double[] apply(double[] array) {
            double[] result = new double[array.length];
            for (int i = 0; i < array.length; i++) {
                result[i] = (array[i] > 0) ? array[i] : array[i] * 0.01;
            }
            return result;
        }

        @Override
        public double[] derive(double[] array) {
            double[] result = new double[array.length];
            for (int i = 0; i < array.length; i++) {
                result[i] = (array[i] > 0) ? 1 : 0.01;
            }
            return result;
        }
    },
    SIGMOID {
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
    },
    TANH {
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
    },
    SOFTMAX {
        @Override
        public double[] apply(double[] array) {
            double[] result = new double[array.length];
            double eSum = 0;

            for (int i = 0; i < array.length; i++) {
                result[i] = Math.exp(array[i]);
                eSum += result[i];
            }

            for (int i = 0; i < array.length; i++) {
                result[i] /= eSum;
                if (Double.isNaN(result[i])) {
                    System.err.println("A certified moment has occurred."); // large outputs cause the sum of e^x in softmax to be infinite, divide by infinite = NaN, then cost is NaN
                    System.exit(1);
                }
            }

            return result;
        }

        @Override
        public double[] derive(double[] array) {
            double[][] result = new double[array.length][array.length];
            double[] softmaxArray = apply(array);

            for (int i = 0; i < array.length; ++i) {
                for (int j = 0; j < array.length; ++j) {
                    if (i == j) {
                        result[i][j] = softmaxArray[i] * (1 - softmaxArray[j]);
                    } else {
                        result[i][j] = softmaxArray[i] * -softmaxArray[j];
                    }
                }
            }

            return Arrays.stream(result).flatMapToDouble(Arrays::stream).toArray();
        }
    };

    public abstract double[] apply(double[] array);

    public abstract double[] derive(double[] array);
}
