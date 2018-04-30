package neuralnet.learn;

import neuralnet.NeuralException;
import neuralnet.NeuralNet;
import neuralnet.data.NeuralDataSet;

public abstract class LearningAlgorithm {
    protected NeuralNet neuralNet;

    public abstract void print();

    public enum LearningMode {ONLINE, BATCH}

    protected enum LearningParadigm {SUPERVISED, UNSUPERVISED}

    protected LearningMode learningMode;

    protected LearningParadigm learningParadigm;


    protected int maxEpochs = 100;
    protected int epoch;
    protected double minOverallError = 0.001;
    protected double learningRate = 0.1;
    protected NeuralDataSet trainingDataSet;
    protected NeuralDataSet testingDataSet;
    protected NeuralDataSet validatingDataSet;
    protected boolean normalization = false;
    public boolean printTraining = false;

    public abstract void train() throws NeuralException;

    public abstract void forward() throws NeuralException;

    public abstract void forward(int i) throws NeuralException;

    public abstract Double calcNewWeight(int layer, int inputNeuron, int neuron) throws NeuralException;

    public abstract Double calcNewWeight(int layer, int input, int neuron, double error) throws NeuralException;

    public NeuralNet getNeuralNet() {
        return neuralNet;
    }

    public void setNeuralNet(NeuralNet neuralNet) {
        this.neuralNet = neuralNet;
    }

    public int getMaxEpochs() {
        return maxEpochs;
    }

    public void setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }

    public int getEpoch() {
        return epoch;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    public double getMinOverallError() {
        return minOverallError;
    }

    public void setMinOverallError(double minOverallError) {
        this.minOverallError = minOverallError;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public NeuralDataSet getTrainingDataSet() {
        return trainingDataSet;
    }

    public void setTrainingDataSet(NeuralDataSet trainingDataSet) {
        this.trainingDataSet = trainingDataSet;
    }

    public NeuralDataSet getTestingDataSet() {
        return testingDataSet;
    }

    public void setTestingDataSet(NeuralDataSet testingDataSet) {
        this.testingDataSet = testingDataSet;
    }

    public NeuralDataSet getValidatingDataSet() {
        return validatingDataSet;
    }

    public void setValidatingDataSet(NeuralDataSet validatingDataSet) {
        this.validatingDataSet = validatingDataSet;
    }

    public boolean isPrintTraining() {
        return printTraining;
    }

    public void setPrintTraining(boolean printTraining) {
        this.printTraining = printTraining;
    }

    public LearningMode getLearningMode() {
        return learningMode;
    }

    public void setLearningMode(LearningMode learningMode) {
        this.learningMode = learningMode;
    }

    public LearningParadigm getLearningParadigm() {
        return learningParadigm;
    }

    public void setLearningParadigm(LearningParadigm learningParadigm) {
        this.learningParadigm = learningParadigm;
    }
}
