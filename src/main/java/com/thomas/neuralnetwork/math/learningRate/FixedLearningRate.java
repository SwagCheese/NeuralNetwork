package com.thomas.neuralnetwork.math.learningRate;

public class FixedLearningRate implements LearningRate {
    private final double learningRate;

    public FixedLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public double get() {
        return learningRate;
    }

    @Override
    public void update(int epoch) {
        // not needed
    }
}
