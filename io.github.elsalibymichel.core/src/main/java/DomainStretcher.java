import java.util.Arrays;

public class DomainStretcher {

  private final double[] stretchFactors;

  public DomainStretcher(double[] multiplicationFactors) {
    this.stretchFactors = multiplicationFactors;
  }

  public static DomainStretcher equalStretcher(int dimension, double stretchFactor) {
    double[] multiplicationFactors = new double[dimension];
    Arrays.fill(multiplicationFactors, stretchFactor);
    return new DomainStretcher(multiplicationFactors);
  }

  public double[] stretch(double[] point) {
    double[] stretchedPoint = new double[point.length];
    for (int i = 0; i < point.length; i++) {
      stretchedPoint[i] = point[i] * stretchFactors[i];
    }
    return stretchedPoint;
  }

  public double[] unStretch(double[] point) {
    double[] unStretchedPoint = new double[point.length];
    for (int i = 0; i < point.length; i++) {
      unStretchedPoint[i] = point[i] / stretchFactors[i];
    }
    return unStretchedPoint;
  }

  public double[] getMultiplicationFactors() {
    return stretchFactors;
  }
}
