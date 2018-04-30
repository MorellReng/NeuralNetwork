package neuralnet;

import neuralnet.math.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) {
        RandomNumberGenerator.seed = 0;

        int numberOfInputs = 2;
        int numberOfOutputs = 1;
        int[] numberOfHiddenNeurons = {3};
        IActivationFunction[] hiddenAcFnc = {new Sigmoid(1.0)};

        Linear outputAcFnc = new Linear(1.0);
        System.out.println("Creating neural network");

        NeuralNet neuralNet = new NeuralNet(numberOfInputs, numberOfOutputs, numberOfHiddenNeurons, hiddenAcFnc, outputAcFnc);

//        double[] neuralInput = {1.5, 0.5};
        ArrayList<Double> neuralInput = new ArrayList<>();
        neuralInput.add(1.5);
        neuralInput.add(0.5);

        double [] neuralOutput;
        neuralNet.setInputs(neuralInput);
        neuralNet.calc();
        neuralOutput = neuralNet.getOutputs();

        neuralNet.print();

        neuralInput.set(0, 1.0);
        neuralInput.set(1, 2.1);

        neuralNet.setInputs(neuralInput);
        neuralNet.calc();
        neuralOutput = neuralNet.getOutputs();

    }
}
