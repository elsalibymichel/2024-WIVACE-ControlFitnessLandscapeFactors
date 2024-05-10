/*-
 * ========================LICENSE_START=================================
 * jsdynsym-buildable
 * %%
 * Copyright (C) 2023 - 2024 Eric Medvet
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================LICENSE_END==================================
 */
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import io.github.ericmedvet.jnb.core.NamedBuilder;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction;
import io.github.ericmedvet.jsdynsym.buildable.builders.NumericalDynamicalSystems;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationEnvironment;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

public class TwoDNavLandscapeCharacterizer {

  private static final Logger L = Logger.getLogger(TwoDNavLandscapeCharacterizer.class.getName());

  record Pair(String environment, String builder) {}

  record Range(double min, double max) {}

  private static final List<String> FITNESS_FUNCTIONS = List.of("ds.e.n.avgD()", "ds.e.n.minD()", "ds.e.n.finalD()");
  private static final List<Pair> PROBLEMS = List.of(
      // BARRIER
      // Plot: x_axis=innerLayerRatio(1, 2, 3, 4, 5) with fixed nOfSensors=7 and Barrier=C_BARRIER
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 1)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 2)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 4)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 5)"),

      // Plot: x_axis=nOfSensors(3, 5, 7, 9, 11) with fixed innerLayerRatio=3 and Barrier=C_BARRIER
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 3)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 5)", "ds.num.mlp(innerLayerRatio = 3)"),
      // new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 9)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 11)", "ds.num.mlp(innerLayerRatio = 3)"),

      // Plot: x_axis=barrier(A_BARRIER, B_BARRIER, C_BARRIER, D_BARRIER, E_BARRIER) with fixed innerLayerRatio=3
      // and nOfSensors=7
      new Pair("ds.e.navigation(arena = A_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = B_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      // new Pair("ds.e.navigation(arena = C_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = D_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = E_BARRIER; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),

      // MAZE
      // Plot: x_axis=innerLayerRatio(1, 2, 3, 4, 5) with fixed nOfSensors=7 and Barrier=C_MAZE
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 1)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 2)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 4)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 5)"),

      // Plot: x_axis=nOfSensors(3, 5, 7, 9, 11) with fixed innerLayerRatio=3 and Barrier=C_MAZE
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 3)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 5)", "ds.num.mlp(innerLayerRatio = 3)"),
      // new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 9)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 11)", "ds.num.mlp(innerLayerRatio = 3)"),

      // Plot: x_axis=barrier(A_MAZE, B_MAZE, C_MAZE, D_MAZE, E_MAZE) with fixed innerLayerRatio=3 and
      // nOfSensors=7
      new Pair("ds.e.navigation(arena = A_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = B_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      // new Pair("ds.e.navigation(arena = C_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = D_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"),
      new Pair("ds.e.navigation(arena = E_MAZE; nOfSensors = 7)", "ds.num.mlp(innerLayerRatio = 3)"));

  private static final String DEFAULT_FORMAT_PATH =
      "LandscapeCharacterizer__an=%b_ar=%b_s=%d_cn=%d_%s_sn=%d_%s_gb=[%.1f-%.1f]__%s.csv";

  public static class Configuration {

    Random random = new Random();

    @Parameter(
        names = {"--seed", "-s"},
        description = "Seed for the random number generator. If not specified, a random seed is used.")
    public long seed = random.nextInt();

    @Parameter(
        names = {"--centersNumber", "-cn"},
        description = "Number of centers in the genotype space to consider.")
    public int centersNumber = 1000;

    @Parameter(
        names = {"--adaptiveNeighbors", "-an"},
        description = "If true, the number of neighbors will be computed as a function of the genotype space dimension.")
    public boolean adaptiveNeighbors = false;

    @Parameter(
        names = {"--neighborsNumber", "-nn"},
        description = "Number of neighbors to consider for each center. Only used if adaptiveNeighbors is false.")
    public int neighborsNumber = 1;

    @Parameter(
        names = {"--neighborsWeight", "-nw"},
        description =
            "This value and the genotype space dimension will determine the number of neighbors to consider for each center. "
                + "Only used if adaptiveNeighbors is true.")
    public double neighborsWeight = 1; // 50

    @Parameter(
        names = {"--adaptiveResolution", "-ar"},
        description = "If true, the sampling resolution will be computed as a function of the genotype space dimension.")
    public boolean adaptiveResolution = false;

    @Parameter(
        names = {"--samplingResoltion", "-sr"},
        description =
            "Sampling resolution to consider for each couple of center and neighbor."
                + "Only used if adaptiveResolution is false.")
    public double samplingResolution = 0.01;

    @Parameter(
        names = {"--resolutionWeight", "-rw"},
        description = "This value and the genotype space dimension will determine the sampling resolution.")
    public double resolutionWeight = 0.35;

    @Parameter(
        names = {"--samplesNumber", "-sn"},
        description = "Number of samples to consider for each couple of center and neighbor.")
    public int samplesNumber = 100; // 60

    @Parameter(
        names = {"--genotypeBounds", "-gb"},
        description = "Bounds for the genotype components.")
    public List<Double> genotypeBoundsList = Arrays.asList(-3.0, 3.0);

    @Parameter(
        names = {"--resultsTarget", "-t"},
        description = "File path where to store the results.")
    public String resultsTarget = DEFAULT_FORMAT_PATH;

    @Parameter(
        names = {"--deltaUpdate", "-du"},
        description = "Delta [sec] update for the progress printer.")
    public int deltaUpdate = 10;

    @Parameter(
        names = {"--help", "-h"},
        description = "Show this help.",
        help = true)
    public boolean help;
  }

  private static final NamedBuilder<Object> BUILDER = NamedBuilder.fromDiscovery();

  public static Range getGenotypeBounds(List<Double> genotypeBoundsList) {
    if (genotypeBoundsList.size() != 2) {
      throw new IllegalArgumentException("GenotypeBounds requires exactly 2 arguments.");
    }
    return new Range(genotypeBoundsList.get(0), genotypeBoundsList.get(1));
  }

  @SuppressWarnings("unchecked")
  private static double[] getFitnessValues(Pair problem, double[] mlpWeights) {
    NavigationEnvironment environment = (NavigationEnvironment) BUILDER.build(problem.environment);
    MultiLayerPerceptron mlp = ((NumericalDynamicalSystems.Builder<MultiLayerPerceptron, ?>)
        BUILDER.build(problem.builder))
        .apply(environment.nOfOutputs(), environment.nOfInputs());
    SingleAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], NavigationEnvironment.State> task =
        SingleAgentTask.fromEnvironment(environment, new double[2], new DoubleRange(0, 60), 1 / 60d);
    mlp.setParams(mlpWeights);
    Simulation.Outcome<SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>> outcome =
        task.simulate(mlp);
    return FITNESS_FUNCTIONS.stream()
        .mapToDouble(s -> ((FormattedNamedFunction<
            Simulation.Outcome<
                SingleAgentTask.Step<double[], double[], NavigationEnvironment.State>>,
            Double>)
            BUILDER.build(s))
            .apply(outcome))
        .toArray();
  }

  private static int computeNeighborNumber(int genotypeSize, boolean adaptive_neighbors_number, double neighbors_weight, int fixed_neighbors_number) {
    if (adaptive_neighbors_number) {
      return (int) Math.ceil(neighbors_weight *  genotypeSize);
    }
    else {
      return fixed_neighbors_number;
    }
  }

  private static double computeSegmentLength(int genotypeSize, boolean adaptive_sampling_rate, double resolution_weight, double fixed_sampling_rate, int samples_number){
    if (adaptive_sampling_rate) {
      return resolution_weight * Math.sqrt(genotypeSize) * samples_number;
    }
    else {
      return fixed_sampling_rate * samples_number;
    }
  }

  @SuppressWarnings("unchecked")
  public static void main(String[] args) throws FileNotFoundException {

    Locale.setDefault(Locale.ROOT);

    // Parse command line options
    Configuration configuration = new Configuration();
    JCommander jc = JCommander.newBuilder().addObject(configuration).build();
    jc.setProgramName(TwoDNavLandscapeCharacterizer.class.getName());
    try {
      jc.parse(args);
    } catch (ParameterException e) {
      e.usage();
      L.severe(String.format("Cannot read command line options: %s", e));
      System.exit(-1);
    } catch (RuntimeException e) {
      L.severe(e.getClass().getSimpleName() + ": " + e.getMessage());
      System.exit(-1);
    }

    // Check help
    if (configuration.help) {
      jc.usage();
      System.exit(0);
    }

    // Initialize the genotype bounds
    final Range genotypeBounds = getGenotypeBounds(configuration.genotypeBoundsList);

    // Check if the resultsTarget is the default one and if so, update it with the experiment values
    if (configuration.resultsTarget.equals(DEFAULT_FORMAT_PATH)) {
      ZonedDateTime timestamp = ZonedDateTime.now(); // Use ZonedDateTime
      DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH.mm.ss");

      // Denpending on the adaptiveNeighbors and adaptiveResolution options, the resultsTarget will be updated
      String neighbors_info_string;
      String resolution_info_string;
      if (configuration.adaptiveNeighbors){
        neighbors_info_string = String.format("nw=%.2f", configuration.neighborsWeight);
      }
      else{
        neighbors_info_string = String.format("nn=%d", configuration.neighborsNumber);
      }
      if (configuration.adaptiveResolution){
        resolution_info_string = String.format("rw=%.2f", configuration.resolutionWeight);
      }
      else{
        resolution_info_string = String.format("sr=%.2f", configuration.samplingResolution);
      }

      // Format the resultsTarget string
      configuration.resultsTarget = String.format(
          DEFAULT_FORMAT_PATH,
          configuration.adaptiveNeighbors,
          configuration.adaptiveResolution,
          configuration.seed,
          configuration.centersNumber,
          neighbors_info_string,
          configuration.samplesNumber,
          resolution_info_string,
          genotypeBounds.min(),
          genotypeBounds.max(),
          timestamp.format(formatter));
    }

    // Create the CSV file
    try (CSVPrinter printer = new CSVPrinter(new FileWriter("csv.txt"), CSVFormat.EXCEL)) {
      printer.printRecord(
          "ENVIRONMENT",
          "BUILDER",
          "CENTER_INDEX",
          "NEIGHBOR_INDEX",
          "SAMPLE_INDEX",
          "SEGMENT_LENGTH",
          "GENOTYPE_SIZE",
          "FITNESS_FUNCTIONS");
    } catch (IOException ex) {
      System.out.printf("Cannot create CSVPrinter: %s%n", ex);
      System.exit(0);
    }

    // Compute the total number of simulations
    int totalSimulations = 0;
    for (Pair problem : PROBLEMS) {
      NavigationEnvironment environment = (NavigationEnvironment) BUILDER.build(problem.environment);
      MultiLayerPerceptron mlp = ((NumericalDynamicalSystems.Builder<MultiLayerPerceptron, ?>)
          BUILDER.build(problem.builder))
          .apply(environment.nOfOutputs(), environment.nOfInputs());
      int genotypeSize = mlp.getParams().length;
      int neighborsNumber = computeNeighborNumber(genotypeSize, configuration.adaptiveNeighbors, configuration.neighborsWeight, configuration.neighborsNumber);
      totalSimulations =
          totalSimulations + configuration.centersNumber * neighborsNumber * configuration.samplesNumber;
    }
    final int totalSimulationsFinal = totalSimulations;

    // Setup the progress printer
    AtomicInteger counterSimulation = new AtomicInteger();
    long initialTime = System.currentTimeMillis();
    Runnable progressPrinterRunnable = () -> {
      System.out.printf("Simulations: %d/%d%n", counterSimulation.get(), totalSimulationsFinal);
      int totalMinutesRemaining = (int) Math.ceil((System.currentTimeMillis() - initialTime)
          / 1000.0
          / counterSimulation.get()
          * (totalSimulationsFinal - counterSimulation.get())
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
    };
    ScheduledExecutorService updatePrinterExecutor = Executors.newScheduledThreadPool(1);
    updatePrinterExecutor.scheduleAtFixedRate(
        progressPrinterRunnable, 0, configuration.deltaUpdate, TimeUnit.SECONDS);

    // Setup the simulation executor and other variables
    PrintStream ps = new PrintStream(configuration.resultsTarget);
    String header = "ENVIRONMENT,BUILDER,CENTER_INDEX,NEIGHBOR_INDEX,SAMPLE_INDEX,SEGMENT_LENGTH,GENOTYPE_SIZE,"
        + String.join(",", FITNESS_FUNCTIONS);
    ps.println(header);
    ExecutorService executorService =
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);
    Random random = new Random(configuration.seed);

    // Start the simulations
    for (Pair problem : PROBLEMS) {

      // Extract the environment and the builder
      NavigationEnvironment environment = (NavigationEnvironment) BUILDER.build(problem.environment);
      MultiLayerPerceptron mlp = ((NumericalDynamicalSystems.Builder<MultiLayerPerceptron, ?>)
          BUILDER.build(problem.builder))
          .apply(environment.nOfOutputs(), environment.nOfInputs());
      int genotypeSize = mlp.getParams().length;

      // Compute the number of neighbors, depending on the adaptiveNeighbors option
      int neighborsNumber = computeNeighborNumber(genotypeSize, configuration.adaptiveNeighbors, configuration.neighborsWeight, configuration.neighborsNumber);
      // Compute the segment length, depending on the adaptiveNeighbors option
      double segmentLength = computeSegmentLength(genotypeSize, configuration.adaptiveResolution, configuration.resolutionWeight, configuration.samplingResolution, configuration.samplesNumber);

      // For each center, compute the central genotype
      for (int center = 0; center < configuration.centersNumber; ++center) {
        double[] centralGenotype = IntStream.range(0, genotypeSize)
            .mapToDouble(i -> genotypeBounds.min()
                + random.nextDouble() * (genotypeBounds.max() - genotypeBounds.min()))
            .toArray();

        // Compute and store the current centralGenotype fitness for all neighbors once and for all
        int finalCenter = center;
        executorService.submit(() -> {
          double[] centralGenotypeFitnessValues = getFitnessValues(problem, centralGenotype);
          for (int n = 0; n < neighborsNumber; ++n) {
            String line = "%s,%s,%d,%d,%d,%.2e,%d,"
                .formatted(
                    problem.environment,
                    problem.builder,
                    finalCenter,
                    n,
                    0,
                    segmentLength,
                    genotypeSize)
                + Arrays.stream(centralGenotypeFitnessValues)
                .mapToObj(value -> String.format("%.5e", value))
                .collect(Collectors.joining(","));
            ps.println(line);
            counterSimulation.getAndIncrement();
          }
        });

        // For each neighbor, sample the corresponding segment and compute the fitness values for each sample
        for (int neighbor = 0; neighbor < neighborsNumber; ++neighbor) {

          // Extracts component with a Gaussian distribution to have a uniform distribution on the n-sphere
          double[] randomVector = IntStream.range(0, genotypeSize)
              .mapToDouble(i -> genotypeBounds.min()
                  + random.nextGaussian() * (genotypeBounds.max() - genotypeBounds.min()))
              .toArray();
          double randomVector_norm = Math.sqrt(Arrays.stream(randomVector)
              .boxed()
              .mapToDouble(element -> element * element)
              .sum());

          // Normalize the random vector and compute the neighbor genotype
          double[] neighborGenotype = IntStream.range(0, genotypeSize)
              .mapToDouble(
                  i -> (randomVector[i] / randomVector_norm) * segmentLength + centralGenotype[i])
              .toArray();

          // Compute the sample step
          double[] sampleStep = IntStream.range(0, genotypeSize)
              .mapToDouble(
                  i -> (neighborGenotype[i] - centralGenotype[i]) / (configuration.samplesNumber - 1))
              .toArray();

          int finalNeighbor = neighbor;
          for (int sample = 1; sample < configuration.samplesNumber; ++sample) {
            int finalSample = sample;
            executorService.submit(() -> {
              StringBuilder line = new StringBuilder();
              line.append("%s,%s,%d,%d,%d,%.2e,%d,"
                  .formatted(
                      problem.environment,
                      problem.builder,
                      finalCenter,
                      finalNeighbor,
                      finalSample,
                      segmentLength,
                      genotypeSize));
              double[] sampleGenotype = Arrays.stream(sampleStep)
                  .boxed()
                  .mapToDouble(s -> s * finalSample)
                  .toArray();
              double[] fitnessValues = getFitnessValues(problem, sampleGenotype);
              line.append(Arrays.stream(fitnessValues)
                  .mapToObj(value -> String.format("%.5e", value))
                  .collect(Collectors.joining(",")));
              ps.println(line);
              counterSimulation.getAndIncrement();
            });
          }
        }
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
