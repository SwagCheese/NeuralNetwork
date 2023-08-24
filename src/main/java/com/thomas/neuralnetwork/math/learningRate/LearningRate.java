package com.thomas.neuralnetwork.math.learningRate;

public interface LearningRate {
    double get();
    void update(int epoch);
}
