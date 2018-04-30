package neuralnet;

import neuralnet.math.RandomNumberGenerator;

import java.util.ArrayList;

public class Neuron {
    protected ArrayList<Double> weights;
    private ArrayList<Double> inputs;
    private double output;
    private double outputBeforeActivation;
    private int numberOfInputs = 0;
    protected double bias = 1.0;
    private IActivationFunction activationFunction;

    public Neuron(int numberOfInputs, IActivationFunction activationFunction) {
        this.numberOfInputs = numberOfInputs;
        this.activationFunction = activationFunction;
        weights = new ArrayList<>(numberOfInputs + 1);
        inputs = new ArrayList<>(numberOfInputs);
    }

    /**
     * Neuron dummy constructor
     */
    public Neuron() {

    }

    /**
     * Neuron constructor
     *
     * @param numberofinputs Number of Inputs
     */
    public Neuron(int numberofinputs) {
        numberOfInputs = numberofinputs;
        weights = new ArrayList<>(numberofinputs + 1);
        inputs = new ArrayList<>(numberofinputs);
    }

    public void init() {
        for (int i = 0; i < numberOfInputs; i++) {
            double newWeight = RandomNumberGenerator.GenerateNext();
            try {
                this.weights.set(i, newWeight);
            } catch (IndexOutOfBoundsException e) {
                this.weights.add(newWeight);
            }
        }
    }

    public void calc() {
        outputBeforeActivation = 0.0;
        if (numberOfInputs > 0) {
            if (inputs != null && weights != null) {
                for (int i = 0; i < numberOfInputs; i++) {
                    outputBeforeActivation += (i == numberOfInputs ? bias : inputs.get(i) * weights.get(i));
                }
            }
        }
        output = activationFunction.calc(outputBeforeActivation);
    }


    public void setActivationFunction(IActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * setInputs
     * Sets a vector of double-precision values to the neuron input
     *
     * @param values vector of values applied at the neuron input
     */
    public void setInputs(double[] values) {
        if (values.length == numberOfInputs) {
            for (int i = 0; i < numberOfInputs; i++) {
                try {
                    inputs.set(i, values[i]);
                } catch (IndexOutOfBoundsException iobe) {
                    inputs.add(values[i]);
                }
            }
        }
    }

    /**
     * updateWeight
     * Method used for updating the weight during learning
     *
     * @param i     ith java position of the weight
     * @param value value to be updated on the weight
     */
    public void updateWeight(int i, double value) {
        if (i >= 0 && i <= numberOfInputs) {
            weights.set(i, value);
        }
    }

    /**
     * setInputs
     * Sets an array of values to the neuron's input
     *
     * @param values
     */
    public void setInputs(ArrayList<Double> values) {
        if (values.size() == numberOfInputs) {
            inputs = values;
        }
    }

    public ArrayList<Double> derivativeBatch(ArrayList<ArrayList<Double>> _input) {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < _input.size(); i++) {
            result.add(0.0);
            Double _outputBeforeActivation = 0.0;
            for (int j = 0; j < numberOfInputs; j++) {
                _outputBeforeActivation += (j == numberOfInputs ? bias : _input.get(i).get(j)) * weights.get(j);
            }
            result.set(i, activationFunction.derivative(_outputBeforeActivation));
        }
        return result;
    }

    public Double getOutput() {
        return output;
    }

    public void setWeights(ArrayList<Double> weights) {
        this.weights = weights;
    }

    public ArrayList<Double> getInputs() {
        return inputs;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getOutputBeforeActivation() {
        return outputBeforeActivation;
    }

    public void setOutputBeforeActivation(double outputBeforeActivation) {
        this.outputBeforeActivation = outputBeforeActivation;
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }

    public void setNumberOfInputs(int numberOfInputs) {
        this.numberOfInputs = numberOfInputs;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public IActivationFunction getActivationFunction() {
        return activationFunction;
    }

    /**
     * getWeights
     * @return Returns the neuron's weights in the form of vector
     */
    public double[] getWeights(){
        double[] weight = new double[numberOfInputs+1];
        for(int i=0;i<=numberOfInputs;i++){
            weight[i]=weights.get(i);
        }
        return weight;
    }

    public Double getWeights(int i){
        return weights.get(i);
    }

    public Double derivative(double[] _input){
        Double _outputBeforeActivation=0.0;
        if(numberOfInputs>0){
            if(weights!=null){
                for(int i=0;i<=numberOfInputs;i++){
                    _outputBeforeActivation+=(i==numberOfInputs?bias:_input[i])*weights.get(i);
                }
            }
        }
        return activationFunction.derivative(_outputBeforeActivation);
    }
}
