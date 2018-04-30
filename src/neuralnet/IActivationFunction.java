package neuralnet;

public interface IActivationFunction {
    double calc(double x);

    /**
     * This is the core method for calculating the activation function's derivative value
     * @param x input value
     * @return returns the derivative result of the activation function given x
     */
    double derivative(double x);

    public enum ActivationFunctionENUM {
        STEP, LINIAR, SIGMOID, HYPERTAN
    }
}
