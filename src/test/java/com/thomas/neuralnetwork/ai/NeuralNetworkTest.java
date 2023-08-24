package com.thomas.neuralnetwork.ai;

import org.junit.Test;

import java.io.*;

import static org.junit.Assert.assertArrayEquals;

public class NeuralNetworkTest {
    @Test
    public void TestFileSaveLoad() {
        NeuralNetwork original = new NeuralNetwork(1000, 10, 100, 1000);

        File file = new File("temp.nnet");
        file.deleteOnExit();

        try (Writer writer = new BufferedWriter(new FileWriter(file))) {
            writer.write(original.toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        NeuralNetwork fromFile = NeuralNetwork.fromFile(file);

        assert fromFile != null;
        assertArrayEquals(original.toArray(), fromFile.toArray());
    }
}