package com.thomas.neuralnetwork.math;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ActivationFunctionTest {
    @Test
    public void testLinearApply() {
        ActivationFunction activationFunction = ActivationFunction.LINEAR;
        double[] input = {1.0, 2.0, 3.0};
        double[] expectedOutput = {1.0, 2.0, 3.0};
        double[] output = activationFunction.apply(input);
        assertArrayEquals(expectedOutput, output, 0.0001);
    }

    @Test
    public void testLinearDerivative() {
        ActivationFunction activationFunction = ActivationFunction.LINEAR;
        double[] input = {1.0, 2.0, 3.0};
        double[] expectedDerivative = {1.0, 1.0, 1.0};
        double[] derivative = activationFunction.derive(input);
        assertArrayEquals(expectedDerivative, derivative, 0.0001);
    }

    @Test
    public void testReluApply() {
        ActivationFunction activationFunction = ActivationFunction.RELU;
        double[] input = {1.0, -2.0, 3.0};
        double[] expectedOutput = {1.0, 0.0, 3.0};
        double[] output = activationFunction.apply(input);
        assertArrayEquals(expectedOutput, output, 0.0001);
    }

    @Test
    public void testReluDerivative() {
        ActivationFunction activationFunction = ActivationFunction.RELU;
        double[] input = {1.0, -2.0, 3.0};
        double[] expectedDerivative = {1.0, 0.0, 1.0};
        double[] derivative = activationFunction.derive(input);
        assertArrayEquals(expectedDerivative, derivative, 0.0001);
    }

    @Test
    public void testSoftmaxApply() {
        ActivationFunction activationFunction = ActivationFunction.SOFTMAX;
        double[] input = {1.0, 2.0, 3.0};
        double[] expectedOutput = {0.09003057317038046, 0.24472847105479764, 0.6652409557748219};
        double[] output = activationFunction.apply(input);
        assertArrayEquals(expectedOutput, output, 0.0001);
        assertEquals(Arrays.stream(output).sum(), 1, 0.0001); // softmax output vector should sum to 1
    }

    @Test
    public void testSoftmaxDerivative() {
        ActivationFunction activationFunction = ActivationFunction.SOFTMAX;
        double[] input = {1.0, 2.0, 3.0};
        double[] expectedDerivative = {0.08192507, -0.02203304, -0.05989202, -0.02203304, 0.18483645, -0.1628034, -0.05989202, -0.1628034, 0.22269543};
        double[] derivative = activationFunction.derive(input);
        assertArrayEquals(expectedDerivative, derivative, 0.0001);
    }
}