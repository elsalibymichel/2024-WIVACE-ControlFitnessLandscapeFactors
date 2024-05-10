import java.util.Arrays;

public class Rastrigin {

  private final int p;
  private final double stretchFactor;

  public Rastrigin(int p, double stretchFactor) {
    this.p = p;
    this.stretchFactor = stretchFactor;
  }

  public double evaluate(double[] vs) {
    return 10d * (double) vs.length
        + Arrays.stream(vs)
        .map(v -> stretchFactor * v)
        .map(v -> v * v - 10 * Math.cos(2 * Math.PI * v))
        .sum();
  }
}
