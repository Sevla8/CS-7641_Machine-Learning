package opt.test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.*;
import java.nio.file.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SingleCrossOver;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.NQueensFitnessFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;

class Analyze_Optimization_Test implements Runnable {

	private class RunValue{
		public double r = 0.0;
		public double t = 0.0;
		RunValue(double result, double time){
			r = result;
			t = time;
		}
	}

	private static Semaphore writeAccess = new Semaphore(1);
	private static Semaphore runPermit = new Semaphore(8);

	/** The number of copies each */
	private static final int COPIES_EACH = 4;
	/** The maximum weight for a single element */
	private static final double MAX_WEIGHT = 50;
	/** The maximum volume for a single element */
	private static final double MAX_VOLUME = 50;

	private Thread t;

	private String problem;
	private String algorithm;
	private int iterations;
	private HashMap<String, Double> params;
	private int N;
	private int T;
	private ConcurrentHashMap<String, String> other_params;
	private int run;

	Analyze_Optimization_Test(
			String problem,
			String algorithm,
			int iterations,
			HashMap<String, Double> params,
			int N,
			int T,
			ConcurrentHashMap<String, String> other_params,
			int run
		) {
		this.problem = problem;
		this.algorithm = algorithm;
		this.iterations = iterations;
		this.params = params;
		this.N = N;
		this.T = T;
		this.other_params = other_params;
		this.run = run;
	}

	private void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
		try {
			writeAccess.acquire();
		} catch (InterruptedException e){
			e.printStackTrace();
		}
		try {
			if (final_result) {
				String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
				String full_path = augmented_output_dir + "/" + file_name;
				Path p = Paths.get(full_path);
				if (Files.notExists(p)) {
					Files.createDirectories(p.getParent());
				}
				PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
				synchronized (pwtr) {
					pwtr.println(results);
					pwtr.close();
				}
			}
			else {
				String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
				Path p = Paths.get(full_path);
				Files.createDirectories(p.getParent());
				try {
					Files.write(p, results.getBytes(), StandardOpenOption.APPEND);
				} catch (NoSuchFileException e){
					Files.write(p, results.getBytes(), StandardOpenOption.CREATE_NEW);
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		writeAccess.release();
	}

	private double[][] buildPoints(int N) {
		Random random = new Random();
		// create the random points
		double[][] points = new double[N][2];
		for (int i = 0; i < points.length; i++) {
			points[i][0] = random.nextDouble();
			points[i][1] = random.nextDouble();
		}
		return points;
	}

	private int[] buildCopies(int N) {
		int[] copies = new int[N];
		Arrays.fill(copies, COPIES_EACH);
		return copies;
	}

	private double[] buildWeights(int N) {
		Random random = new Random();
		double[] weights = new double[N];
		for (int i = 0; i < N; i++) {
			weights[i] = random.nextDouble() * MAX_WEIGHT;
		}
		return weights;
	}

	private double[] buildVolumes(int N) {
		Random random = new Random();
		double[] volumes = new double[N];
		for (int i = 0; i < N; i++) {
			volumes[i] = random.nextDouble() * MAX_VOLUME;
		}
		return volumes;
	}

	public void run() {
		try {
			//wait for a permit to run
			runPermit.acquire();

			EvaluationFunction ef = null;
			Distribution odd = null;
			NeighborFunction nf = null;
			MutationFunction mf = null;
			CrossoverFunction cf = null;
			Distribution df = null;
			int[] ranges;
			double knapsackVolume = MAX_VOLUME * N * COPIES_EACH * .4;

			switch (this.problem) {
				case "count_ones":
					ranges = new int[N];
					Arrays.fill(ranges, 2);
					ef = new CountOnesEvaluationFunction();
					odd = new DiscreteUniformDistribution(ranges);
					nf = new DiscreteChangeOneNeighbor(ranges);
					mf = new DiscreteChangeOneMutation(ranges);
					cf = new UniformCrossOver();
					df = new DiscreteDependencyTree(.1, ranges);
					break;
				case "four_peaks":
					ranges = new int[N];
					Arrays.fill(ranges, 2);
					ef = new FourPeaksEvaluationFunction(T);
					odd = new DiscreteUniformDistribution(ranges);
					nf = new DiscreteChangeOneNeighbor(ranges);
					mf = new DiscreteChangeOneMutation(ranges);
					cf = new SingleCrossOver();
					df = new DiscreteDependencyTree(.1, ranges);
					break;
				case "continuous_peaks":
					ranges = new int[N];
					Arrays.fill(ranges, 2);
					ef = new ContinuousPeaksEvaluationFunction(T);
					odd = new DiscreteUniformDistribution(ranges);
					nf = new DiscreteChangeOneNeighbor(ranges);
					mf = new DiscreteChangeOneMutation(ranges);
					cf = new SingleCrossOver();
					df = new DiscreteDependencyTree(.1, ranges);
					break;
				case "flip_flop":
					ranges = new int[N];
					Arrays.fill(ranges, 2);
					ef = new FlipFlopEvaluationFunction();
					odd = new DiscreteUniformDistribution(ranges);
					nf = new DiscreteChangeOneNeighbor(ranges);
					mf = new DiscreteChangeOneMutation(ranges);
					cf = new SingleCrossOver();
					df = new DiscreteDependencyTree(.1, ranges);
					break;
				case "tsp":
					ranges = new int[N];
					Arrays.fill(ranges, N);
					ef = new TravelingSalesmanRouteEvaluationFunction(buildPoints(N));
					odd = new DiscretePermutationDistribution(N);
					nf = new SwapNeighbor();
					mf = new SwapMutation(); // Does the same thing
					cf = new TravelingSalesmanCrossOver((TravelingSalesmanEvaluationFunction) ef);
					df = new DiscreteDependencyTree(.1, ranges);
				case "nqueens":
					ranges = new int[N];
					Arrays.fill(ranges, N);
					ef = new NQueensFitnessFunction();
					odd = new DiscretePermutationDistribution(N);
					nf = new SwapNeighbor();
					mf = new SwapMutation();
					cf = new SingleCrossOver();
					df = new DiscreteDependencyTree(.1);
				case "knapsack":
					ranges = new int[N];
					Arrays.fill(ranges, COPIES_EACH + 1);
					ef = new KnapsackEvaluationFunction(buildWeights(N), buildVolumes(N), knapsackVolume, buildCopies(N));
					odd = new DiscreteUniformDistribution(ranges);
					nf = new DiscreteChangeOneNeighbor(ranges);
					mf = new DiscreteChangeOneMutation(ranges);
					cf = new UniformCrossOver();
					df = new DiscreteDependencyTree(.1, ranges);
			}

			HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
			GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
			ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

			LinkedList<RunValue> resultList = new LinkedList<>();
			LinkedList<Double> optimaList = new LinkedList<>();

			double optimal_value = -1;
			double start = System.nanoTime();

			switch (this.algorithm) {
				case "RHC":
					RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
					for (int i = 0; i <= this.iterations; i++) {
						resultList.add(new RunValue(rhc.train(), System.nanoTime() - start));
					}
					optimal_value = ef.value(rhc.getOptimal());
					for (int i = 0; i < this.N; i++) {
						optimaList.add(rhc.getOptimal().getData().get(i));
					}
					break;

				case "SA":
					SimulatedAnnealing sa = new SimulatedAnnealing(
							params.get("SA_initial_temperature"),
							params.get("SA_cooling_factor"),
							hcp
					);
					for (int i = 0; i <= this.iterations; i++) {
						resultList.add(new RunValue(sa.train(), System.nanoTime() - start));
					}
					optimal_value = ef.value(sa.getOptimal());
					for (int i = 0; i < this.N; i++) {
						optimaList.add(sa.getOptimal().getData().get(i));
					}
					break;

				case "GA":
					StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(
							params.get("GA_population").intValue(),
							params.get("GA_mate_number").intValue(),
							params.get("GA_mutate_number").intValue(),
							gap
					);
					for (int i = 0; i <= this.iterations; i++) {
						resultList.add(new RunValue(ga.train(), System.nanoTime() - start));
					}
					optimal_value = ef.value(ga.getOptimal());
					for (int i = 0; i < this.N; i++) {
						optimaList.add(ga.getOptimal().getData().get(i));
					}
					break;

				case "MIMIC":
					MIMIC mimic = new MIMIC(
							params.get("MIMIC_samples").intValue(),
							params.get("MIMIC_to_keep").intValue(),
							pop
					);
					for (int i = 0; i <= this.iterations; i++) {
						resultList.add(new RunValue(mimic.train(), System.nanoTime() - start));
					}
					optimal_value = ef.value(mimic.getOptimal());
					for (int i = 0; i < this.N; i++) {
						optimaList.add(mimic.getOptimal().getData().get(i));
					}
					break;
			}

			double end = System.nanoTime();
			double timeSeconds = (end - start) / Math.pow(10,9);

			StringBuilder baseLine = new StringBuilder();
			baseLine.append(this.problem);
			baseLine.append(",");
			baseLine.append(this.algorithm);
			baseLine.append(",");
			baseLine.append(this.N);
			baseLine.append(",");
			baseLine.append(this.T);
			baseLine.append(",");
			String[] paramList = {
				"SA_initial_temperature",
				"SA_cooling_factor",
				"GA_population",
				"GA_mate_number",
				"GA_mutate_number",
				"MIMIC_samples",
				"MIMIC_to_keep"
			};
			for(String p : paramList){
				if(params.containsKey(p))
					baseLine.append(params.get(p));
				else
					baseLine.append(0);
				baseLine.append(",");
			}
			String lineStart = baseLine.toString();
			StringBuilder sb = new StringBuilder();
			int i = 0;
			for (RunValue r : resultList){
				sb.append(lineStart);
				sb.append(r.r);
				sb.append(",");
				sb.append(r.t);
				sb.append(",");
				sb.append(i++);
				sb.append("\n");
			}

			write_output_to_file(this.other_params.get("output_folder"), "problems_results.csv", sb.toString(), false);

			StringBuilder generalLine = new StringBuilder();
			generalLine.append(lineStart);
			generalLine.append(timeSeconds);
			generalLine.append(",");
			generalLine.append(iterations);
			generalLine.append(",");
			generalLine.append(optimal_value);
			generalLine.append("\n");

			write_output_to_file(this.other_params.get("output_folder"), "problems.csv", generalLine.toString(), false);
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		runPermit.release();
	}

	public void start () {
		if (t == null)
		{
			t = new Thread (this);
			t.start ();
		}
	}
}

public class OptimizationTest {

	public static void main(String[] args) {

		ConcurrentHashMap<String, String> other_params = new ConcurrentHashMap<>();
		other_params.put("output_folder","stats");
		int num_runs = 10;

		HashMap<String, LinkedList<Double[]>> param_map = new HashMap<>();

		LinkedList<Double[]> RHCparmList = new LinkedList<>();
		RHCparmList.add(new Double[] {0., 0., 0., 0., 0., 0., 0.});
		param_map.put("RHC", RHCparmList);

		LinkedList<Double[]> SAparmList = new LinkedList<>();
		for(double it=50.0 ; it <= 200.0 ; it += 50.0){
			for(double cooling=0.5 ; cooling < 1.0 ; cooling += 0.05 ){
				SAparmList.add(new Double[] {it, cooling, 0., 0., 0., 0., 0.});
			}
		}
		param_map.put("SA", SAparmList);

		LinkedList<Double[]> GAparmList = new LinkedList<>();
		for(double pop=100.0 ; pop <= 600.0 ; pop += 100.0){
			for(double mate=50.0 ; mate < pop ; mate += 100.0 ){
				for(double mutate=10 ; mutate <= mate/5.0 ; mutate += 20.0) {
					GAparmList.add(new Double[]{0., 0., pop, mate, mutate, 0., 0.});
				}
			}
		}
		param_map.put("GA", GAparmList);

		LinkedList<Double[]> MIMICparmList = new LinkedList<>();
		for(double sample=50.0 ; sample <= 400.0 ; sample += 100.0){
			for(double to_keep=10.0 ; to_keep < sample/2. ; to_keep += 50.0 ){
				MIMICparmList.add(new Double[]{0., 0., 0., 0., 0., sample, to_keep});
			}
		}
		param_map.put("MIMIC", MIMICparmList);

		int numberOfParamCombinations = (MIMICparmList.size()+RHCparmList.size()+GAparmList.size()+SAparmList.size());
		System.out.println("Testing for "+numberOfParamCombinations+" combinations of parameters");

		//Four Peaks Test

		int[] N = {50};
		int[] iterations = {5000, 5000, 5000, 5000};
		String[] algorithms = {"RHC", "SA", "GA", "MIMIC"};
		for (int i = 0; i < algorithms.length; i++) {
			 for( Double[] param : param_map.get(algorithms[i])){
				 HashMap<String, Double> params = new HashMap<>();
				 params.put("SA_initial_temperature",param[0]);
				 params.put("SA_cooling_factor",param[1]);
				 params.put("GA_population",param[2]);
				 params.put("GA_mate_number",param[3]);
				 params.put("GA_mutate_number",param[4]);
				 params.put("MIMIC_samples",param[5]);
				 params.put("MIMIC_to_keep",param[6]);
				 for (int j = 0; j < N.length; j++) {
					 for (int l = 0; l < num_runs; l++) {
						 new Analyze_Optimization_Test(
								 "four_peaks",
								 algorithms[i],
								 iterations[i],
								 params,
								 N[j],
								 N[j]/5,
								 other_params,
								 l
						 ).start();
					 }
				 }
			 }

		}

		//Traveling Salesman Problem

		N = new int[]{50};
		iterations = new int[]{5000, 5000, 5000, 5000};
		for (int i = 0; i < algorithms.length; i++) {
			for( Double[] param : param_map.get(algorithms[i])){
				HashMap<String, Double> params = new HashMap<>();
				params.put("SA_initial_temperature", param[0]);
				params.put("SA_cooling_factor", param[1]);
				params.put("GA_population", param[2]);
				params.put("GA_mate_number", param[3]);
				params.put("GA_mutate_number", param[4]);
				params.put("MIMIC_samples", param[5]);
				params.put("MIMIC_to_keep", param[6]);
				for (int j = 0; j < N.length; j++) {
					for (int l = 0; l < num_runs; l++) {
						new Analyze_Optimization_Test(
								"tsp",
								algorithms[i],
								iterations[i],
								params,
								N[j],
								N[j] / 5,
								other_params,
								l
						).start();
					}
				}
			}
		}

		// Flip Flop

		N = new int[]{50};
		iterations = new int[]{5000, 5000, 5000, 5000};
		for (int i = 0; i < algorithms.length; i++) {
			for( Double[] param : param_map.get(algorithms[i])){
				HashMap<String, Double> params = new HashMap<>();
				params.put("SA_initial_temperature", param[0]);
				params.put("SA_cooling_factor", param[1]);
				params.put("GA_population", param[2]);
				params.put("GA_mate_number", param[3]);
				params.put("GA_mutate_number", param[4]);
				params.put("MIMIC_samples", param[5]);
				params.put("MIMIC_to_keep", param[6]);
				for (int j = 0; j < N.length; j++) {
					for (int l = 0; l < num_runs; l++) {
						new Analyze_Optimization_Test(
								"flip_flop",
								algorithms[i],
								iterations[i],
								params,
								N[j],
								N[j] / 5,
								other_params,
								l
						).start();
					}
				}
			}
		}
	}
}

