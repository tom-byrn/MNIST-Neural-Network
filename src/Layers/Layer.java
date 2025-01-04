package Layers;

import java.util.List;

public abstract class Layer {

    protected Layer _nextLayer;
    protected Layer _previousLayer;

    //Convolutional layers and max pull layer takes lists of matrix inputs
    public abstract double[] getOutput(List<double[][]> input);

    //Polymorphic overloading so that a vector can also be taken
    public abstract double[] getOutput(double[] input);

    //Takes the derivative dLoss/dOutput as the input (also uses overloading again)
    public abstract void backPropagation(List<double[]> dLdO);
    public abstract void backPropagation(double[] dLdO);
}