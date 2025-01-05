package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{

    private long SEED; //Used for setting weights
    private final double leak = 0.01;

    private double[][] _weights;
    private int _inLength;
    private int _outLength;
    private double _learningRate;

    private double[] lastZ;
    private double[] lastX;

    public FullyConnectedLayer(int _inLength, int _outLength, long SEED, double _learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this._learningRate = _learningRate;
        this.SEED = SEED;

        //Each output is the sum of each input * weight
        //There is a separate weight for each interaction, so the size of the weights matrix is going to be the number of inputs * outputs
        _weights = new double[_inLength][_outLength];
        setRandomWeights();
    }

    public double[] fullyConnectedForwardPass(double[] input){

        lastX = input;

        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        for(int i = 0; i < _inLength; i++){
            for(int j = 0; j < _outLength; j++){

                try {
                    z[j] += input[i] * _weights[i][j];
                } catch(IndexOutOfBoundsException e){
                    System.out.println("input length: " + input.length);
                    System.out.println("Length of loop for i (_inLength): " + _inLength + "\t\tValue of i: " + i);
                    System.out.println("Length of loop for j/size of z/size of out (_outLength): " + _outLength + "\t\tValue of j: " + j);
                    throw new RuntimeException(e);
                }
            }
        }

        lastZ = z;

        for(int j = 0; j < _outLength; j++){
            out[j] = relu(z[j]);
        }


        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardpass = fullyConnectedForwardPass(input);

        if(_nextLayer != null){
            return _nextLayer.getOutput(forwardpass);
        } else{
            return forwardpass; //If on the last layer in the chain, just return what you have since there is not another layer to pass it to
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        //Using chain rule to get dLength/dWeight
        double dOdz;
        double dZdW;
        double dLdW;
        double dZdX;

        double[] dLdX = new double[_inLength];

        for(int k = 0; k < _inLength; k++){

            double dLdX_sum = 0;

            for(int j = 0; j < _outLength; j++){

                dOdz = derivativeRelu(lastZ[j]);
                dZdW = lastX[k];
                dZdX = _weights[k][j];

                dLdW = dLdO[j] * dOdz * dZdW;

                _weights[k][j] -= dLdW * _learningRate;

                dLdX_sum += dLdO[j] * dOdz * dZdX;
            }

            dLdX[k] = dLdX_sum; //Equal to the sum of X has contributed to the loss of all of our outputs

        }
        if(_previousLayer != null) {
            _previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outLength;
    }

    public void setRandomWeights(){
        Random random = new Random(SEED);

        for(int i = 0; i < _inLength; i++){
            for(int j = 0; j < _outLength; j++){
                _weights[i][j] = random.nextGaussian();//A gaussian is a number around 0 so can be negative or positive (Randomness is in a gaussian curve / normal distribution)
            }
        }
    }

    public double relu(double input){
        if(input <= 0) {
            return 0;
        } else{
            return input;
        }
    }

    public double derivativeRelu(double input){
        if(input <= 0) {
            return leak;
        } else{
            return 1;
        }
    }
}
