package com.thomas.neuralnetwork.math.learningRate;

public class CyclicLearningRate implements LearningRate {
    private int batchesSinceRestart;
    private int nextCycleRestart;

    private double maxLearningRate;
    private final double minLearningRate;
    private int cycleLength;
    private final double cycleLengthMultiplier;
    private final double learningRateDecay;
    private final double batchesPerEpoch;

    public CyclicLearningRate(double minLearningRate, double maxLearningRate, int cycleLength, double cycleLengthMultiplier, double learningRateDecay, int batchesPerEpoch) {
        this.minLearningRate = minLearningRate;
        this.maxLearningRate = maxLearningRate;
        this.cycleLength = cycleLength;
        this.cycleLengthMultiplier = cycleLengthMultiplier;
        this.learningRateDecay = learningRateDecay;
        this.batchesPerEpoch = batchesPerEpoch;
    }

    @Override
    public double get() {
        double fractionToRestart = (double) ++batchesSinceRestart / (batchesPerEpoch * cycleLength);
        return minLearningRate + 0.5 * (maxLearningRate - minLearningRate) * (1 + Math.cos(fractionToRestart * Math.PI));
    }

    @Override
    public void update(int epoch) {
        if (epoch + 1 == nextCycleRestart) {
            batchesSinceRestart = 0;

            cycleLength = (int) ((double)cycleLength * cycleLengthMultiplier);
            nextCycleRestart += cycleLength;

            maxLearningRate *= learningRateDecay;
        }
    }
}
