package neuralnet;

import java.util.ArrayList;

public abstract class NeuralLayer {

    /**
     * Number of Neurons in this Layer
     */
    protected int numberOfNeuronsInLayer;
    /**
     * Array of Neurons of this Layer
     */
    protected ArrayList<Neuron> neurons;

    /**
     * Activation Function of this Layer
     */
    protected IActivationFunction activationFunction;

    /**
     * Previous Layer that feeds values to this Layer
     */
    protected NeuralLayer previousLayer;
    /**
     * Next Layer which this Layer will feed values to
     */
    protected NeuralLayer nextLayer;

    /**
     * Array of input values that are fed to this Layer
     */
    protected ArrayList<Double> inputs;
    /**
     * Array of output values this Layer will produce
     */
    protected ArrayList<Double> output;

    /**
     * Number of Inputs this Layer can receive
     */
    protected int numberOfInputs;


    protected NeuralNet neuralNet;
    /**
     * NeuralLayer constructor
     *
     * @param numberofneurons Number of Neurons in this Layer
     * @see NeuralLayer
     */
    public NeuralLayer(int numberofneurons){
        this.numberOfNeuronsInLayer=numberofneurons;
        neurons = new ArrayList<>(numberofneurons);
        output = new ArrayList<>(numberofneurons);
    }

    /**
     * NeuralLayer constructor
     *
     * @param numberofneurons Number of Neurons in this Layer
     * @param iaf Activation Function for all neurons in this Layer
     * @see NeuralLayer
     */
    public NeuralLayer(int numberofneurons,IActivationFunction iaf){
        this.numberOfNeuronsInLayer=numberofneurons;
        this.activationFunction=iaf;
        neurons = new ArrayList<>(numberofneurons);
        output = new ArrayList<>(numberofneurons);
    }

    protected void init() {
        for (int i = 0; i < numberOfNeuronsInLayer; i++) {
            try {
                neurons.get(i).setActivationFunction(activationFunction);
                neurons.get(i).init();
            } catch (IndexOutOfBoundsException e) {
                neurons.add(new Neuron(numberOfInputs, activationFunction));
                neurons.get(i).init();
            }
        }
    }

    protected void calc() {
        for (int i = 0; i < numberOfNeuronsInLayer; i++) {
            neurons.get(i).setInputs(this.inputs);
            neurons.get(i).calc();
            try {
                output.set(i, neurons.get(i).getOutput());
            } catch (IndexOutOfBoundsException e){
                output.add(neurons.get(i).getOutput());
            }
        }
    }

    /**
     * getNumberOfNeuronsInLayer
     * @return Returns the number of neurons in this layer
     */
    public int getNumberOfNeuronsInLayer(){
        return numberOfNeuronsInLayer;
    }

    /**
     * getListOfNeurons
     * @return Returns the whole array of neurons of this layer
     */
    public ArrayList<Neuron> getListOfNeurons(){
        return neurons;
    }

    /**
     * getPreviousLayer
     * @return Returns the reference to the previous layer
     */
    protected NeuralLayer getPreviousLayer(){
        return previousLayer;
    }

    /**
     * getNextLayer
     * @return Returns the reference to the next layer
     */
    protected NeuralLayer getNextLayer(){
        return nextLayer;
    }

    /**
     * setPreviousLayer
     * @param layer Sets the reference to the previous layer
     */
    public void setPreviousLayer(NeuralLayer layer){
        previousLayer=layer;
    }

    /**
     * setNextLayer
     * @param layer Sets the reference to the next layer
     */
    protected void setNextLayer(NeuralLayer layer){
        nextLayer=layer;
    }

    /**
     * getOutputs
     * @return Returns the array of this layer's outputs
     */
    protected ArrayList<Double> getOutputs(){
        return output;
    }

    /**
     * getNeuron
     * @param i java index of the neuron
     * @return Returns the Neuron at the i-th java position in the layer
     */
    public Neuron getNeuron(int i){
        return neurons.get(i);
    }

    /**
     * setNeuron
     * Sets an already created Neuron at this layer's input
     * @param i java index where the Neuron will be placed
     * @param _neuron Neuron to be inserted or placed in the layer
     */
    protected void setNeuron(int i, Neuron _neuron){
        try{
            this.neurons.set(i, _neuron);
        }
        catch(IndexOutOfBoundsException iobe){
            this.neurons.add(_neuron);
        }
    }

    /**
     * setInputs
     * Sets an array of real values to this layer's input
     * @param inputs array of real values to be fed into this layer's input
     * @see
     */
    protected void setInputs(ArrayList<Double> inputs){
        this.numberOfInputs=inputs.size();
        this.inputs=inputs;
    }

}
