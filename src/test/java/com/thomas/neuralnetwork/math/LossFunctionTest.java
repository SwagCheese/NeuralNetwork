package com.thomas.neuralnetwork.math;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LossFunctionTest {
    @Test
    public void testMeanSquaredError() {
        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        double[] actual = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        double[] predicted = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        double expected = 33.0;
        double result = lossFunction.calculate(actual, predicted);
        assertEquals(expected, result, 0.0001);
    }

    @Test
    public void testMeanAbsoluteError() {
        LossFunction lossFunction = LossFunction.MEAN_ABSOLUTE_ERROR;
        double[] actual = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        double[] predicted = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        double expected = 5;
        double result = lossFunction.calculate(actual, predicted);
        assertEquals(expected, result, 0.0001);
    }

    @Test
    public void testCrossEntropyError() {
        LossFunction lossFunction = LossFunction.CROSS_ENTROPY_ERROR;
        double[] actual = {0.0, 1.0};
        double[] predicted = {0.2, 0.8};
        double expected = 0.2231; // calculate the expected value manually
        double result = lossFunction.calculate(actual, predicted);
        assertEquals(expected, result, 0.0001);
    }
}