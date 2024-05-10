import io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationEnvironment;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RastriginLandscapeCharacterizer {

  private static final int CENTERS_NUMBER = 1000;

  private static final double NEIGHBORS_WEIGHT = 2;
  private static final boolean ADAPTIVE_NEIGHBORS_NUMBER = false;
  private static final int FIXED_NEIGHBORS_NUMBER = 1; // Only used if ADAPTIVE_NEIGHBORS_NUMBER is false

  private static final double RESOLUTION_WEIGHT = 0.01;
  private static final boolean ADAPTIVE_SAMPLING_RATE = true;
  private static final double FIXED_SAMPLING_RATE = 0.01; // Only used if ADAPTIVE_SAMPLING_RATE is false

  private static final int SAMPLES_NUMBER = 100;
  private static final Double[] GENOTYPE_BOUNDS = new Double[]{-3d, 3d};
  private static final int DELTA_UPDATE = 10;
  private static final long SEED = 1;
  private static final String RESULTS_TARGET = "results_dynamicSampling_multiple_p.csv";


  private static final List<Double> STRETCH_FACTORS = List.of(1d, 2d, 3d, 4d, 5d);
  private static final List<Integer> GENOTYPE_SIZES = List.of(2, 10, 18, 26);

  private static final List<String> FITNESS_FUNCTIONS = List.of("rastrigin");


  private static double[] getFitnessValues(double stretchFactor, int genotypeSize, double[] genotype) {
    return new double[]{new Rastrigin(genotypeSize, stretchFactor).evaluate(genotype)};
  }

  private static int computeNeighborNumber(int genotypeSize) {
    if (ADAPTIVE_NEIGHBORS_NUMBER) {
      return (int) Math.ceil(NEIGHBORS_WEIGHT * Math.sqrt(genotypeSize));
    }
    else {
      return FIXED_NEIGHBORS_NUMBER;
    }
  }

  private static double computeSegmentLength(int genotypeSize) {
    if (ADAPTIVE_SAMPLING_RATE) {
      return RESOLUTION_WEIGHT * Math.sqrt(genotypeSize) * SAMPLES_NUMBER;
    }
    else {
      return FIXED_SAMPLING_RATE * SAMPLES_NUMBER;
    }
  }

  public static void main(String[] args) throws FileNotFoundException {

    final int totalCombinations = STRETCH_FACTORS.size() * GENOTYPE_SIZES.size() * FITNESS_FUNCTIONS.size();

    // Setup the progress printer
    AtomicInteger counterCombination = new AtomicInteger();
    long initialTime = System.currentTimeMillis();
    Runnable progressPrinterRunnable = () -> {
      System.out.printf("Simulations: %d/%d%n", counterCombination.get(), totalCombinations);
    };
    ScheduledExecutorService updatePrinterExecutor = Executors.newScheduledThreadPool(1);
    updatePrinterExecutor.scheduleAtFixedRate(
        progressPrinterRunnable, 0, DELTA_UPDATE, TimeUnit.SECONDS);

    // Setup the simulation executor and other variables
    PrintStream ps = new PrintStream(RESULTS_TARGET);
    String header = "STRETCH_FACTOR,CENTER_INDEX,NEIGHBOR_INDEX,SAMPLE_INDEX,SEGMENT_LENGTH,GENOTYPE_SIZE,"
        + String.join(",", FITNESS_FUNCTIONS);
    ps.println(header);
    ExecutorService executorService =
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);
    Random random = new Random(SEED);

    // Start the simulations
    for (Double stretchFactor : STRETCH_FACTORS) {
      for (Integer genotypeLength : GENOTYPE_SIZES) {

        // Compute the number of neighbors and the segment length
        int neighborsNumber = computeNeighborNumber(genotypeLength);
        double segmentLength = computeSegmentLength(genotypeLength);

        // For each center, compute the central genotype
        for (int center = 0; center < CENTERS_NUMBER; ++center) {
          double[] centralGenotype = IntStream.range(0, genotypeLength)
              .mapToDouble(i -> GENOTYPE_BOUNDS[0]
                  + random.nextDouble() * (GENOTYPE_BOUNDS[1] - GENOTYPE_BOUNDS[0]))
              .toArray();

          // Compute and store the current centralGenotype fitness for all neighbors once and for all
          int finalCenter = center;
          executorService.submit(() -> {
            double[] centralGenotypeFitnessValues = getFitnessValues(stretchFactor, genotypeLength, centralGenotype);
            for (int n = 0; n < neighborsNumber; ++n) {
              String line = "%.2e,%d,%d,%d,%.2e,%d,"
                  .formatted(
                      stretchFactor,
                      finalCenter,
                      n,
                      0,
                      segmentLength,
                      genotypeLength)
                  + Arrays.stream(centralGenotypeFitnessValues)
                  .mapToObj(value -> String.format("%.5e", value))
                  .collect(Collectors.joining(","));
              ps.println(line);
            }
          });

          // For each neighbor, sample the corresponding segment and compute the fitness values for each sample
          for (int neighbor = 0; neighbor < neighborsNumber; ++neighbor) {

            // Extracts component with a Gaussian distribution to have a uniform distribution on the n-sphere
            double[] randomVector = IntStream.range(0, genotypeLength)
                .mapToDouble(i -> GENOTYPE_BOUNDS[0]
                    + random.nextGaussian() * (GENOTYPE_BOUNDS[1] - GENOTYPE_BOUNDS[0]))
                .toArray();
            double randomVector_norm = Math.sqrt(Arrays.stream(randomVector)
                .boxed()
                .mapToDouble(element -> element * element)
                .sum());

            // Normalize the random vector and compute the neighbor genotype
            double[] neighborGenotype = IntStream.range(0, genotypeLength)
                .mapToDouble(
                    i -> (randomVector[i] / randomVector_norm) * segmentLength + centralGenotype[i])
                .toArray();

            // Compute the sample step
            double[] sampleStep = IntStream.range(0, genotypeLength)
                .mapToDouble(
                    i -> (neighborGenotype[i] - centralGenotype[i]) / (SAMPLES_NUMBER - 1))
                .toArray();

            int finalNeighbor = neighbor;
            for (int sample = 1; sample < SAMPLES_NUMBER; ++sample) {
              int finalSample = sample;
              executorService.submit(() -> {
                StringBuilder line = new StringBuilder();
                line.append("%.2e,%d,%d,%d,%.2e,%d,"
                    .formatted(
                        stretchFactor,
                        finalCenter,
                        finalNeighbor,
                        finalSample,
                        segmentLength,
                        genotypeLength));
                double[] sampleGenotype = Arrays.stream(sampleStep)
                    .boxed()
                    .mapToDouble(s -> s * finalSample)
                    .toArray();
                double[] fitnessValues = getFitnessValues(stretchFactor, genotypeLength, sampleGenotype);
                line.append(Arrays.stream(fitnessValues)
                    .mapToObj(value -> String.format("%.5e", value))
                    .collect(Collectors.joining(",")));
                ps.println(line);
              });
            }
          }
        }

        counterCombination.getAndIncrement();
      }
    }

    executorService.shutdown();
    boolean terminated = false;
    while (!terminated) {
      try {
        terminated = executorService.awaitTermination(1, TimeUnit.SECONDS);
      } catch (InterruptedException e) {
        // ignore
      }
    }
    updatePrinterExecutor.shutdown();
    System.out.println("Done");
    ps.close();
  }

}
