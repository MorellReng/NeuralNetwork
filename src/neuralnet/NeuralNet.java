package neuralnet;

import java.util.ArrayList;

public class NeuralNet {
    private InputLayer inputLayer;
    private ArrayList<HiddenLayer> hiddenLayer;
    private OutputLayer outputLayer;
    private int numberOfHiddenLayers;
    private int numberOfInputs;
    private int numberOfOutputs;
    private ArrayList<Double> input;
    private ArrayList<Double> output;

    public void calc() {
        inputLayer.setInputs(input);
        inputLayer.calc();
        for (int i = 0; i < numberOfHiddenLayers; i++) {
            HiddenLayer hl = hiddenLayer.get(i);
            hl.setInputs(hl.getPreviousLayer().getOutputs());
            hl.calc();
        }
        outputLayer.setInputs(outputLayer.getPreviousLayer().getOutputs());
        outputLayer.calc();
        this.output = outputLayer.getOutputs();
    }

    public NeuralNet(int numberOfInputs, int numberOfOutputs, int[] numberOfHiddenNeurons, IActivationFunction[] hiddenAcFnc, IActivationFunction outputAcFnc) {
        this.numberOfHiddenLayers = numberOfHiddenNeurons.length;
        this.numberOfInputs = numberOfInputs;
        this.numberOfOutputs = numberOfOutputs;

        if (numberOfHiddenLayers == hiddenAcFnc.length) {
            input = new ArrayList<>(numberOfInputs);
            inputLayer = new InputLayer(numberOfInputs);
            if (numberOfHiddenLayers > 0) {
                hiddenLayer = new ArrayList<>(numberOfHiddenLayers);
            }
            for (int i = 0; i < numberOfHiddenLayers; i++) {
                if (i == 0) {
                    try {
                        hiddenLayer.set(i, new HiddenLayer(numberOfHiddenNeurons[i],
                                hiddenAcFnc[i],
                                inputLayer.getNumberOfNeuronsInLayer()));
                    } catch (IndexOutOfBoundsException iobe) {
                        hiddenLayer.add(new HiddenLayer(numberOfHiddenNeurons[i],
                                hiddenAcFnc[i],
                                inputLayer.getNumberOfNeuronsInLayer()));
                    }
                    inputLayer.setNextLayer(hiddenLayer.get(i));
                } else {
                    try {
                        hiddenLayer.set(i, new HiddenLayer(numberOfHiddenNeurons[i],
                                hiddenAcFnc[i], hiddenLayer.get(i - 1)
                                .getNumberOfNeuronsInLayer()
                        ));
                    } catch (IndexOutOfBoundsException iobe) {
                        hiddenLayer.add(new HiddenLayer(numberOfHiddenNeurons[i],
                                hiddenAcFnc[i], hiddenLayer.get(i - 1)
                                .getNumberOfNeuronsInLayer()
                        ));
                    }
                    hiddenLayer.get(i - 1).setNextLayer(hiddenLayer.get(i));
                }
            }
            if (numberOfHiddenLayers > 0) {
                outputLayer = new OutputLayer(numberOfOutputs, outputAcFnc,
                        hiddenLayer.get(numberOfHiddenLayers - 1)
                                .getNumberOfNeuronsInLayer()
                );
                hiddenLayer.get(numberOfHiddenLayers - 1).setNextLayer(outputLayer);
            } else {
                outputLayer = new OutputLayer(numberOfOutputs, outputAcFnc,
                        numberOfOutputs);
                inputLayer.setNextLayer(outputLayer);
            }
        }

    }

    /**
     * print
     * Method to print the neural network information
     */
    public void print(){
        System.out.println("Neural Network: "+this.toString());
        System.out.println("\tInputs:"+String.valueOf(this.numberOfInputs));
        System.out.println("\tOutputs:"+String.valueOf(this.numberOfOutputs));
        System.out.println("\tHidden Layers: "+String.valueOf(numberOfHiddenLayers));
        for(int i=0;i<numberOfHiddenLayers;i++){
            System.out.println("\t\tHidden Layer "+
                    String.valueOf(i)+": "+
                    String.valueOf(this.hiddenLayer.get(i)
                            .numberOfNeuronsInLayer)+" Neurons");
        }

    }

    /**
     * setInputs
     * Feeds an array of real values to the neural network's inputs
     * @param inputs Array of real values to be fed into the neural inputs
     */
    public void setInputs(ArrayList<Double> inputs){
        if(inputs.size()==numberOfInputs){
            this.input=inputs;
        }
    }

    /**
     * setInputs
     * Sets a vector of double-precision values into the neural network inputs
     * @param inputs vector of values to be fed into the neural inputs
     */
    public void setInputs(double[] inputs){
        if(inputs.length==numberOfInputs){
            for(int i=0;i<numberOfInputs;i++){
                try{
                    input.set(i, inputs[i]);
                }
                catch(IndexOutOfBoundsException iobe){
                    input.add(inputs[i]);
                }
            }
        }
    }

    /**
     * getOutputs
     * @return Returns the neural outputs in the form of vector
     */
    public double[] getOutputs(){
        double[] _outputs = new double[numberOfOutputs];
        for(int i=0;i<numberOfOutputs;i++){
            _outputs[i]=output.get(i);
        }
        return _outputs;
    }


    public InputLayer getInputLayer() {
        return inputLayer;
    }

    public void setInputLayer(InputLayer inputLayer) {
        this.inputLayer = inputLayer;
    }

    public ArrayList<HiddenLayer> getHiddenLayer() {
        return hiddenLayer;
    }

    public void setHiddenLayer(ArrayList<HiddenLayer> hiddenLayer) {
        this.hiddenLayer = hiddenLayer;
    }

    public OutputLayer getOutputLayer() {
        return outputLayer;
    }

    public void setOutputLayer(OutputLayer outputLayer) {
        this.outputLayer = outputLayer;
    }

    public int getNumberOfHiddenLayers() {
        return numberOfHiddenLayers;
    }

    public void setNumberOfHiddenLayers(int numberOfHiddenLayers) {
        this.numberOfHiddenLayers = numberOfHiddenLayers;
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }

    public void setNumberOfInputs(int numberOfInputs) {
        this.numberOfInputs = numberOfInputs;
    }

    public int getNumberOfOutputs() {
        return numberOfOutputs;
    }

    public void setNumberOfOutputs(int numberOfOutputs) {
        this.numberOfOutputs = numberOfOutputs;
    }

    public ArrayList<Double> getInput() {
        return input;
    }

    public void setInput(ArrayList<Double> input) {
        this.input = input;
    }

    public ArrayList<Double> getOutput() {
        return output;
    }

    public void setOutput(ArrayList<Double> output) {
        this.output = output;
    }

    public HiddenLayer getHiddenLayer(int l) {
        return hiddenLayer.get(l);
    }

    public double[] getInputs(){
        double[] result=new double[numberOfInputs];
        for(int i=0;i<numberOfInputs;i++){
            result[i]=input.get(i);
        }
        return result;
    }

    /**
     * getInput
     * @param i ith java position at the input
     * @return Returns the ith java input
     */
    public double getInput(int i){
        return input.get(i);
    }
}
