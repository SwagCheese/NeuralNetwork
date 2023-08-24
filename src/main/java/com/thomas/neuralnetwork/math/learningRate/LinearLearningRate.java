package com.thomas.neuralnetwork.math.learningRate;

public class LinearLearningRate implements LearningRate {
    private double learningRate;
    private final double increment;

    public LinearLearningRate(double initialLearningRate, double increment) {
        this.learningRate = initialLearningRate;
        this.increment = increment;
    }
    @Override
    public double get() {
        return learningRate;
    }

    @Override
    public void update(int epoch) {
        learningRate += increment;
    }
}
