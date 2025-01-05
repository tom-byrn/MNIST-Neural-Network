package layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {

    public Layer get_nextLayer() {
        return _nextLayer;
    }

    public void set_nextLayer(Layer _nextLayer) {
        this._nextLayer = _nextLayer;
    }

    public Layer get_previousLayer() {
        return _previousLayer;
    }

    public void set_previousLayer(Layer _previousLayer) {
        this._previousLayer = _previousLayer;
    }

    protected Layer _nextLayer;
    protected Layer _previousLayer;

    //Convolutional layers and max pull layer takes lists of matrix inputs
    public abstract double[] getOutput(List<double[][]> input);

    //Polymorphic overloading so that a vector can also be taken
    public abstract double[] getOutput(double[] input);

    //Takes the derivative dLoss/dOutput as the input (also uses overloading again)
    public abstract void backPropagation(List<double[][]> dLdO);
    public abstract void backPropagation(double[] dLdO);

    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getOutputElements();


    //Convert matrix to vector
    public double[] matrixToVector(List<double[][]> input){

        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length*rows*cols];

        //linearise the matrix (Goes through every single value and adds it to array)
        int i = 0;
        for(int l = 0; l < length; l++){
            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    vector[i] = input.get(l)[r][c];
                    i++;
                }
            }
        }
        return vector;
    }


    //Convert vector to matrix
    List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols){

        List<double[][]> out = new ArrayList<>();

        for(int i = 0; i < length; i++) {

            double[][] matrix = new double[rows][cols];

            for (int l = 0; l < length; l++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        matrix[r][c] = input[i];
                        i++;
                    }
                }
            }
            out.add(matrix);
        }
    return out;
    }

}