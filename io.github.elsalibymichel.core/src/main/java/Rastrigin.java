import java.util.Arrays;

public class Rastrigin {

  private final double[] genotype;
  private final double stretchFactor;

  public Rastrigin(double[] genotype, double stretchFactor) {
    this.genotype = genotype;
    this.stretchFactor = stretchFactor;
  }

  public double evaluate() {
    return 10d * (double) this.genotype.length
        + Arrays.stream(this.genotype)
        .map(v -> stretchFactor * v)
        .map(v -> v * v - 10 * Math.cos(2 * Math.PI * v))
        .sum();
  }
}
