import io.github.ericmedvet.jnb.core.NamedBuilder;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction;
import io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationEnvironment;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;
import org.apache.commons.csv.CSVFormat;

import java.io.*;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.io.Reader;
import java.nio.file.Paths;

import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class BestGenotypesLandscapeCharacterizer {

  record Pair(String environment, String builder) {
  }

  private static final int NEIGHBORS_NUMBER = 100;
  private static final int SAMPLES_NUMBER = 100;
  private static final long SEED = 1;
  private static final String RESULTS_TARGET = "neurons_dynamic_optimalPoint_characterized.csv";

  private static final boolean ADAPTIVE_SAMPLING_RATE = true;
  private static final double RESOLUTION_WEIGHT = 0.007;  // 0.007 is 0.01/sqrt(2)
  private static final double FIXED_SAMPLING_RATE = 0.01; // Only used if ADAPTIVE_SAMPLING_RATE is false

  private static final String FILE_PATH = "/home/melsalib/Desktop/IntellijProjects/paper-ruggedness/io.github.elsalibymichel.core/src/data/sensors_onlyLast.csv";

  private static final NamedBuilder<Object> BUILDER = NamedBuilder.fromDiscovery();

  private static AtomicInteger JOBS_DONE = new AtomicInteger(0);
  private static AtomicInteger TOTAL_JOBS = new AtomicInteger(0);
  private static long INITIAL_TIME = System.currentTimeMillis();

  private static double computeSegmentLength(int genotypeSize) {
    if (ADAPTIVE_SAMPLING_RATE) {
      return RESOLUTION_WEIGHT * Math.sqrt(genotypeSize) * SAMPLES_NUMBER;
    } else {
      return FIXED_SAMPLING_RATE * SAMPLES_NUMBER;
    }
  }

  @SuppressWarnings("unchecked")
  private static double getFitnessValues(Pair problem, double[] mlpWeights, String fitnessFunction) {
    // System.out.println("Problem: " + problem.environment);
    NavigationEnvironment environment = (NavigationEnvironment) BUILDER.build(problem.environment);
    MultiLayerPerceptron mlp = ((NumericalDynamicalSystems.Builder<MultiLayerPerceptron, ?>)
        BUILDER.build(problem.builder))
        .apply(environment.nOfOutputs(), environment.nOfInputs());
    // System.out.println("A");
    SingleAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], NavigationEnvironment.State> task =
        SingleAgentTask.fromEnvironment(environment, new double[2], new DoubleRange(0, 60), 1 / 60d);
    mlp.setParams(mlpWeights);
    Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>> outcome =
        task.simulate(mlp);
    // System.out.println("B");
    try {
      double fitnessValue = ((FormattedNamedFunction<
          Simulation.Outcome<
              SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>>,
          Double>)
          BUILDER.build(fitnessFunction))
          .apply(outcome);
      // Increment the counter of jobs done
      JOBS_DONE.getAndIncrement();
      return fitnessValue;
    } catch (Throwable t) {
      t.printStackTrace();
      System.exit(-1);
      return -1;
    }
  }

  @SuppressWarnings("unchecked")
  public static void main(String[] args) throws IOException {

    Locale.setDefault(Locale.ROOT);
    String filePath;
    try {
      filePath = args[0];
    } catch (Exception e) {
      System.out.println("No file path provided, default path will be used.\n");
      filePath = FILE_PATH;
    }

    try {
      Files.deleteIfExists(Paths.get(RESULTS_TARGET));
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(-2);
    }
    try (PrintStream ps = new PrintStream(RESULTS_TARGET)) {
      String header = "ENVIRONMENT,BUILDER,CENTER_INDEX,NEIGHBOR_INDEX,SAMPLE_INDEX,SEGMENT_LENGTH,GENOTYPE_SIZE,FITNESS_FUNCTION,FITNESS_VALUE";
      ps.println(header);
      // double segmentLength = SAMPLES_NUMBER * FIXED_SAMPLING_RATE;

      // Set up the simulation executor and other variables
      ExecutorService executorService =
          Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);

      Random random = new Random(SEED);

      try (
          Reader reader = Files.newBufferedReader(Paths.get(filePath));
          CSVParser csvParser = CSVFormat.Builder.create().setDelimiter(";").build().parse(reader);
      ) {
        int i = 0;
        for (CSVRecord csvRecord : csvParser) {
          if (i == 0) {
            i = i + 1;
            continue;
          }
          int center = Integer.parseInt(csvRecord.get(0)); // seed
          String fitnessFunction = csvRecord.get(3); // fitness function
          double[] centralGenotype = Arrays.stream(csvRecord.get(16).replaceAll("[\\[\\]]", "").split(","))
              .mapToDouble(Double::valueOf)
              .toArray();
          // int nSensors = Integer.parseInt(csvRecord.get(1)); // number of sensors
          // Pair problem = new Pair(String.format("ds.e.navigation(arena = A_BARRIER; nOfSensors = %d)", nSensors), "ds.num.mlp(innerLayerRatio = 3)");
          // int nNeurons = Integer.parseInt(csvRecord.get(1)); // number of sensors
          // Pair problem = new Pair("ds.e.navigation(arena = A_BARRIER; nOfSensors = 7)", String.format("ds.num.mlp(innerLayerRatio = %d)", nNeurons));
          int nNeuronLayers = Integer.parseInt(csvRecord.get(1)); // number of sensors
          Pair problem = new Pair("ds.e.navigation(arena = A_BARRIER; nOfSensors = 7)", String.format("ds.num.mlp(nOfInnerLayers = %d)", nNeuronLayers));


          // Extract the environment and the builder and check if they are correctly parsed
          NavigationEnvironment environment = (NavigationEnvironment) BUILDER.build(problem.environment);
          MultiLayerPerceptron mlp = ((NumericalDynamicalSystems.Builder<MultiLayerPerceptron, ?>)
              BUILDER.build(problem.builder))
              .apply(environment.nOfOutputs(), environment.nOfInputs());
          int genotypeSize = mlp.getParams().length;
          if (genotypeSize != centralGenotype.length) {
            System.out.println("Genotype size mismatch");
            System.exit(1);
          }

          double segmentLength = computeSegmentLength(genotypeSize);

          // Compute and store the current centralGenotype fitness for all neighbors once and for all
          // Increment the total counter for the central genotype
          TOTAL_JOBS.getAndIncrement();
          executorService.submit(() -> {
            double centralGenotypeFitnessValues = getFitnessValues(problem, centralGenotype, fitnessFunction);
            for (int n = 0; n < NEIGHBORS_NUMBER; ++n) {
              String line = "%s,%s,%d,%d,%d,%.2e,%d,%s,%.5e"
                  .formatted(
                      problem.environment,
                      problem.builder,
                      center,
                      n,
                      0,
                      segmentLength,
                      genotypeSize,
                      fitnessFunction,
                      centralGenotypeFitnessValues);
              ps.println(line);
            }
          });

          // For each neighbor, sample the corresponding segment and compute the fitness values for each sample
          for (int neighbor = 0; neighbor < NEIGHBORS_NUMBER; ++neighbor) {
            // Extracts component with a Gaussian distribution to have a uniform distribution on the n-sphere
            double[] randomVector = IntStream.range(0, genotypeSize)
                .mapToDouble(s -> random.nextGaussian())
                .toArray();
            // Compute the norm of the random vector
            double randomVector_norm = Math.sqrt(Arrays.stream(randomVector)
                .boxed()
                .mapToDouble(element -> element * element)
                .sum());
            // Normalize the random vector and compute the neighbor genotype
            double[] neighborGenotype = IntStream.range(0, genotypeSize)
                .mapToDouble(
                    s -> (randomVector[s] / randomVector_norm) * segmentLength + centralGenotype[s])
                .toArray();
            // Compute the sample step
            double[] sampleStep = IntStream.range(0, genotypeSize)
                .mapToDouble(
                    s -> (neighborGenotype[s] - centralGenotype[s]) / (SAMPLES_NUMBER - 1))
                .toArray();

            TOTAL_JOBS.getAndIncrement();
            for (int sample = 1; sample < SAMPLES_NUMBER; ++sample) {
              // Increment the total counter for each other sample except for the central one
              TOTAL_JOBS.getAndIncrement();
              final int finalNeighbor = neighbor;
              final int finalSample = sample;
              executorService.submit(() -> {
                double[] sampleGenotype = Arrays.stream(sampleStep)
                    .boxed()
                    .mapToDouble(s -> s * finalSample)
                    .toArray();
                double fitnessValue = getFitnessValues(problem, sampleGenotype, fitnessFunction);
                String line = "%s,%s,%d,%d,%d,%.2e,%d,%s,%.5e"
                    .formatted(
                        problem.environment,
                        problem.builder,
                        center,
                        finalNeighbor,
                        finalSample,
                        segmentLength,
                        genotypeSize,
                        fitnessFunction,
                        fitnessValue);
                ps.println(line);
              });
            }
          }
        }
      }

      executorService.shutdown();
      boolean terminated = false;
      while (!terminated) {
        try {
          System.out.println("Jobs done: " + JOBS_DONE.get() + "/" + TOTAL_JOBS.get());
          int totalMinutesRemaining = (int) Math.ceil((System.currentTimeMillis() - INITIAL_TIME)
              / 1000.0
              / JOBS_DONE.get()
              * (TOTAL_JOBS.get() - JOBS_DONE.get())
              / 60);
          int hours = totalMinutesRemaining / 60;
          int minutes = totalMinutesRemaining % 60;
          int days = hours / 24;
          if (days > 0) {
            System.out.printf("Remaining time estimate: %4d d  %2d h  %2d min%n", days, hours % 24, minutes);
          } else if (hours > 0) {
            System.out.printf("Remaining time estimate: %2d h  %2d min%n", hours, minutes);
          } else {
            System.out.printf("Remaining time estimate: %2d min%n", minutes);
          }

          terminated = executorService.awaitTermination(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
          // ignore
        }
      }
      System.out.println("Done");
    }
  }
}
