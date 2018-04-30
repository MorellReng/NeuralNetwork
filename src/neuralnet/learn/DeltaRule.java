package neuralnet.learn;

import neuralnet.HiddenLayer;
import neuralnet.NeuralException;
import neuralnet.Neuron;
import neuralnet.OutputLayer;

import java.util.ArrayList;

public class DeltaRule extends LearningAlgorithm {
    public ArrayList<ArrayList<Double>> error;
    public ArrayList<Double> generalError;
    public ArrayList<Double> overallError;
    public double overallGeneralError;
    public double degreeGeneralError = 2.0;
    public double degreeOverallError = 0.0;

    public enum ErrorMeasurement {SimpleError, SquareError, NDegreeError, MSE}

    public ErrorMeasurement generalErrorMeasurement = ErrorMeasurement.SquareError;
    public ErrorMeasurement overallErrorMeasurement = ErrorMeasurement.MSE;

    private int currentRecord = 0;
    private ArrayList<ArrayList<ArrayList<Double>>> newWeights;

    @Override
    public void train() throws NeuralException {
        switch (learningMode) {

            case BATCH:
                epoch = 0;
                forward();
                while (epoch < maxEpochs && overallGeneralError > minOverallError) {
                    epoch++;
                    for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
                        for (int i = 0; i < neuralNet.getNumberOfInputs(); i++) {
                            newWeights.get(0).get(j).set(i, calcNewWeight(0, i, j));
                        }
                    }
                    applyNewWeights();
                    forward();
                }
                break;

            case ONLINE:
                epoch = 0;
                int k = 0;
                currentRecord = 0;
                forward(k);

                while (epoch < maxEpochs && overallGeneralError > minOverallError) {
                    for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
                        for (int i = 0; i < neuralNet.getNumberOfInputs(); i++) {
                            newWeights.get(0).get(j).set(i, calcNewWeight(0, i, j));
                        }
                    }
                    applyNewWeights();
                    currentRecord = ++k;
                    if (k >= trainingDataSet.numberOfRecords) {
                        k = 0;
                        currentRecord = 0;
                        epoch++;
                    }
                    forward(k);
                }
                break;
        }
    }

    @Override
    public void forward(int i) throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0) {
            throw new NeuralException("Delta rule can be used only with single "
                    + "layer neural network");
        } else {
            neuralNet.setInputs(trainingDataSet.getArrayInputRecord(i, normalization));
            neuralNet.calc();
            trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs(), normalization);
            generalError.set(i,
                    generalError(
                            trainingDataSet.getArrayTargetOutputRecord(i, normalization)
                            , trainingDataSet.getArrayNeuralOutputRecord(i, normalization)));
            for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
                overallError.set(j,
                        overallError(trainingDataSet
                                        .getIthTargetOutputArrayList(j, normalization)
                                , trainingDataSet
                                        .getIthNeuralOutputArrayList(j, normalization)));
                error.get(i).set(j
                        , simpleError(trainingDataSet
                                        .getIthTargetOutputArrayList(j, normalization).get(i)
                                , trainingDataSet.getIthNeuralOutputArrayList(j, normalization)
                                        .get(i)));
            }
            overallGeneralError = overallGeneralErrorArrayList(
                    trainingDataSet.getArrayTargetOutputData(normalization)
                    , trainingDataSet.getArrayNeuralOutputData(normalization));
            //simpleError=simpleErrorEach.get(i);
        }
    }

    @Override
    public void forward() throws NeuralException {
        if (neuralNet.getNumberOfHiddenLayers() > 0) {
            throw new NeuralException("Delta rule can be used only with single"
                    + " layer neural network");
        } else {
            for (int i = 0; i < trainingDataSet.numberOfRecords; i++) {
                neuralNet.setInputs(trainingDataSet.getInputRecord(i, normalization));
                neuralNet.calc();
                trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs(), normalization);
                generalError.set(i,
                        generalError(
                                trainingDataSet.getArrayTargetOutputRecord(i, normalization)
                                , trainingDataSet.getArrayNeuralOutputRecord(i, normalization)));
                for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
                    error.get(i).set(j
                            , simpleError(trainingDataSet
                                            .getArrayTargetOutputRecord(i, normalization).get(j)
                                    , trainingDataSet.getArrayNeuralOutputRecord(i, normalization)
                                            .get(j)));
                }
            }
            for (int j = 0; j < neuralNet.getNumberOfOutputs(); j++) {
                overallError.set(j,
                        overallError(trainingDataSet
                                        .getIthTargetOutputArrayList(j, normalization)
                                , trainingDataSet
                                        .getIthNeuralOutputArrayList(j, normalization)));
            }
            overallGeneralError = overallGeneralErrorArrayList(
                    trainingDataSet.getArrayTargetOutputData(normalization)
                    , trainingDataSet.getArrayNeuralOutputData(normalization));
            //simpleError=simpleErrorEach.get(trainingDataSet.numberOfRecords-1);
        }
    }

    @Override
    public Double calcNewWeight(int layer, int input, int neuron) throws NeuralException {
        Double deltaWeight = learningRate;
        Neuron currentNeuron = neuralNet.getOutputLayer().getNeuron(neuron);
        switch (learningMode) {

            case BATCH:
                ArrayList<Double> deriviationResult = currentNeuron.derivativeBatch(trainingDataSet.getArrayInputData());
                ArrayList<Double> ithInput;
                if (input < currentNeuron.getNumberOfInputs()) {// Weights
                    ithInput = trainingDataSet.getIthInputArrayList(input);
                } else {//bias
                    ithInput = new ArrayList<>();
                    for (int i = 0; i < trainingDataSet.numberOfRecords; i++) {
                        ithInput.add(1.0);
                    }
                }
                Double multDerivResultIthInput = 0.0; // Dot product
                for (int i = 0; i < trainingDataSet.numberOfRecords; i++) {
                    multDerivResultIthInput += error.get(i).get(neuron) * deriviationResult.get(i) * ithInput.get(i);
                }
                deltaWeight *= multDerivResultIthInput;

                break;

            case ONLINE:
                deltaWeight *= error.get(currentRecord).get(neuron);
                deltaWeight *= currentNeuron.derivative(neuralNet.getInputs());
                if (input < currentNeuron.getNumberOfInputs()) {
                    deltaWeight *= neuralNet.getInput(input);
                }

                break;

        }
        return currentNeuron.getWeights(input) + deltaWeight;
    }

    @Override
    public Double calcNewWeight(int layer, int input, int neuron, double error) throws NeuralException {
        return null;
    }

    @SuppressWarnings("Duplicates")
    public void applyNewWeights() {
        int numberOfHiddenLayers = this.neuralNet.getNumberOfHiddenLayers();
        for (int l = 0; l <= numberOfHiddenLayers; l++) {
            int numberOfNeuronsInLayer, numberOfInputsInNeuron;
            if (l < numberOfHiddenLayers) {
                HiddenLayer hl = this.neuralNet.getHiddenLayer(l);
                numberOfNeuronsInLayer = hl.getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = hl.getNeuron(j).getNumberOfInputs();
                    for (int i = 0; i <= numberOfInputsInNeuron; i++) {
                        double newWeight = this.newWeights.get(l).get(j).get(i);
                        hl.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            } else {
                OutputLayer ol = this.neuralNet.getOutputLayer();
                numberOfNeuronsInLayer = ol.getNumberOfNeuronsInLayer();
                for (int j = 0; j < numberOfNeuronsInLayer; j++) {
                    numberOfInputsInNeuron = ol.getNeuron(j).getNumberOfInputs();

                    for (int i = 0; i <= numberOfInputsInNeuron; i++) {
                        double newWeight = this.newWeights.get(l).get(j).get(i);
                        ol.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            }
        }
    }

    public Double simpleError(Double YT, Double Y) {
        return YT - Y;
    }

    public Double squareError(Double YT, Double Y) {
        return (1.0 / 2.0) * Math.pow(YT - Y, 2.0);
    }

    public Double overallGeneralErrorArrayList(ArrayList<ArrayList<Double>> YT, ArrayList<ArrayList<Double>> Y) {
        int N = YT.size();
        int Ny = YT.get(0).size();
        Double result = 0.0;
        for (int i = 0; i < N; i++) {
            Double resultY = 0.0;
            for (int j = 0; j < Ny; j++) {
                resultY += Math.pow(YT.get(i).get(j) - Y.get(i).get(j), degreeGeneralError);
            }
            if (generalErrorMeasurement == ErrorMeasurement.MSE)
                result += Math.pow((1.0 / Ny) * resultY, degreeOverallError);
            else
                result += Math.pow((1.0 / degreeGeneralError) * resultY, degreeOverallError);
        }
        if (overallErrorMeasurement == ErrorMeasurement.MSE)
            result *= (1.0 / N);
        else
            result *= (1.0 / degreeOverallError);
        return result;
    }

    public Double generalError(ArrayList<Double> YT, ArrayList<Double> Y) {
        int Ny = YT.size();
        Double result = 0.0;
        for (int i = 0; i < Ny; i++) {
            result += Math.pow(YT.get(i) - Y.get(i), degreeGeneralError);
        }
        if (generalErrorMeasurement == ErrorMeasurement.MSE)
            result *= (1.0 / Ny);
        else
            result *= (1.0 / degreeGeneralError);
        return result;
    }

    public Double overallError(ArrayList<Double> YT, ArrayList<Double> Y) {
        int N = YT.size();
        Double result = 0.0;
        for (int i = 0; i < N; i++) {
            result += Math.pow(YT.get(i) - Y.get(i), degreeOverallError);
        }
        if (overallErrorMeasurement == ErrorMeasurement.MSE)
            result *= (1.0 / N);
        else
            result *= (1.0 / degreeOverallError);
        return result;
    }
}
