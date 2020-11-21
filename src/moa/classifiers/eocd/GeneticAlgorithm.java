package moa.classifiers.eocd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

// Essential metaheuristics page 37
public class GeneticAlgorithm {

	public final StopCondition stopCondition;

	public final Problem problem;

	public final Random rand;

	private final int numBestSolves;

	private final List<Solve> bestSolves;
	
	protected int sizePopulation = 30;

	protected int numElitism = 3;

	protected Double rateLocus = 0.5; // [0.3,0.95]

	protected Double rateCrossover = 0.9;

	protected int numRounds = 2;

	private List<Solve> population = new ArrayList<>();

	public GeneticAlgorithm(Problem problem, StopCondition stopCondition, Random rand) {
		this.problem = problem;
		this.stopCondition = stopCondition;
		this.rand = rand;
		this.numBestSolves = 1;
		this.bestSolves = new ArrayList<>(numBestSolves + 10);
	}

	protected void run() {
		//System.out.println("GA");

		population = newPopulation(sizePopulation);
		//System.out.println("START POP");
		for (Solve solve : population) {
			evaluate(solve);
			recordBestSolve(solve);
			//System.out.println("\t" + solve);
		}
		//System.out.println();
		//System.out.println("BEST SOLVE " + this.bestSolve());

		//System.out.println(stopCondition);
		while (stopCondition.isRunning()) {
			List<Solve> offsprings = breed(population);
			//System.out.println("OFFSPRINGS");
			for (Solve solve : offsprings) {
				evaluate(solve);
				recordBestSolve(solve);
				//System.out.println("\t" + solve);
			}
			//System.out.println();
			//System.out.println("BEST SOLVE " + this.bestSolve());
			join(population, offsprings, sizePopulation);
			stopCondition.iteration();

			//System.out.println("NEW POP");
			//Collections.sort(population);
			//for (Solve solve : population) {
			//	System.out.println("\t" + solve);
			//}
			//System.out.println();
			//System.out.println("BEST SOLVE " + this.bestSolve());

			//System.out.println(stopCondition);
		}
	}

	protected List<Solve> breed(List<Solve> population) {
		List<Solve> offsprings = new ArrayList<>();
		List<Solve> parents = select(population, sizePopulation);
		for (int i = 0; i + 1 < parents.size(); i += 2) {
			List<Solve> selectedParents = new ArrayList<>();
			for (int j = 0; j < 2; j++) {
				selectedParents.add(parents.get(j + i));
			}
			List<Solve> off = crossover(selectedParents);
			for (Solve solve : off) {
				mutation(solve);
			}
			offsprings.addAll(off);
		}
		return offsprings;
	}

	public void execute() {
		stopCondition.start();
		run();
		stopCondition.stop();
	}

	public void evaluate(Solve solve) {
		if (stopCondition.isRunning()) {
			solve.repair();
			problem.evaluate(solve);
			solve.evaluated(true);
			stopCondition.evaluation();
		} else {
			solve.evaluated(false);
		}
	}

	protected void recordBestSolve(Solve solve) {
		if (!solve.evaluated() || !solve.isValid()) {
			return;
		}
		if (bestSolves.isEmpty()) {
			bestSolves.add(solve);
			stopCondition.update();
		} else {
			int index = 0;
			while (index < bestSolves.size() && solve.compareTo(bestSolves.get(index)) > 0) {
				index += 1;
			}
			if (index < numBestSolves) {
				Solve place = index < bestSolves.size() ? bestSolves.get(index) : bestSolves.get(bestSolves.size() - 1);
				if (!solve.equals(place) || !solve.equalsEncode(place)) {
					bestSolves.add(index, solve);
					if (index == 0) {
						stopCondition.update();
					}
					if (bestSolves.size() > numBestSolves) {
						bestSolves.remove(bestSolves.size() - 1);
					}
				}
			}
		}
	}

	public Solve bestSolve() {
		Solve min = null;
		for (Solve solve : bestSolves) {
			if (min == null || solve.compareTo(min) < 0) {
				min = solve;
			}
		}
		return min;
	}

	public List<Solve> bestSolves() {
		return bestSolves;
	}

	public List<Solve> population() {
		return population;
	}

	public int populationSize() {
		return sizePopulation;
	}

	protected List<Solve> newPopulation(int size) {
		List<Solve> solves = new ArrayList<>();
		if (!problem.configurations.isEmpty()) {
			for (int i = 0; i < size / 2; i++) {
				int index = rand.nextInt(problem.configurations.size());
				EnsembleMemberConfiguration conf = problem.configurations.get(index);
				Solve solve = (Solve) problem.createEmptySolve();
				conf.configure(solve);
				solves.add(solve);
			}
		}
		for (int i = 0; i < size - solves.size(); i++) {
			Solve solve = problem.createEmptySolve();
			for (int j = 0; j < solve.encode.length; i++) {
				solve.encode[j] = rand.nextDouble();
			}
			solves.add(solve);
		}
		return solves;
	}

	protected void join(List<Solve> population, List<Solve> offsprings, int sizePop) {
		Collection<Solve> finalPop = new ArrayList<>();
		Collections.sort(offsprings);

		if (numElitism > 0) {
			Collections.sort(population);
			int size = Math.min(numElitism, population.size());
			for (int i = 0; i < size; i++) {
				Solve solve = population.remove(0);
				if (finalPop.isEmpty() || solve.compareTo(offsprings.get(0)) < 0) {
					finalPop.add(solve);
				}
			}
		}

		while (finalPop.size() < sizePop && !offsprings.isEmpty()) {
			finalPop.add(offsprings.remove(0));
		}

		population.clear();
		population.addAll(finalPop);
	}

	protected List<Solve> crossover(List<Solve> parents) { // uniform
		Solve pa = parents.get(0);
		Solve pb = parents.get(1);
		Solve oa = null;
		Solve ob = null;
		if (rand.nextDouble() < rateCrossover) {
			oa = problem.createEmptySolve();
			ob = problem.createEmptySolve();
			apply(pa.encode, pb.encode, oa.encode, ob.encode, rateLocus, rand);
		} else {
			oa = pa.clone();
			ob = pb.clone();
		}
		return Arrays.asList(oa, ob);
	}

	private static void apply(double[] pa, double[] pb, double[] oa, double[] ob, double rate, Random rand) {
		for (int i = 0; i < pa.length; i++) {
			if (rand.nextDouble() < rate) {
				oa[i] = pa[i];
				ob[i] = pb[i];
			} else {
				oa[i] = pb[i];
				ob[i] = pa[i];
			}
		}
	}

	protected void mutation(Solve solve) { // bit flip proportional uniform
		int num = generatePoissonNumber(1, rand); // 1/l
		for (int i = 0; i < num; i++) {
			int index = rand.nextInt(problem.sizeEncode);
			solve.encode[index] = solve.encode[index] + rand.nextGaussian();
			if (solve.encode[index] < 0) {
				solve.encode[index] += 1.0;
			} else if (solve.encode[index] > 1.0) {
				solve.encode[index] -= 1.0;
			}
		}
	}

	private static int generatePoissonNumber(double lambda, Random rand) { // knuth algorithm
		double threshold = Math.exp(-lambda);
		int events = 0;
		double acc = 1;
		do {
			events += 1;
			acc *= rand.nextDouble();
		} while (acc > threshold);
		return events - 1;
	}

	protected List<Solve> select(List<Solve> population, int size) {
		List<Solve> parents = new ArrayList<>();
		for (int i = 0; i < size; i++) {
			int index = rand.nextInt(population.size());
			Solve best = population.get(index);
			for (int round = 1; round < numRounds; round++) {
				index = rand.nextInt(population.size());
				Solve solve = population.get(index);
				if (solve.compareTo(best) < 0) {
					best = solve;
				}
			}
			parents.add(best);
		}
		return parents;
	}
}
