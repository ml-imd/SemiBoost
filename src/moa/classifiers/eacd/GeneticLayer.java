package moa.classifiers.eacd;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.AbstractClassifier;

public class GeneticLayer {

	protected int maxNumBerOfGenerations;

	protected Instances instances;

	protected boolean implicitReset;

	protected Type[] startTypes;

	protected Type[] finalPopulation;

	protected Random rand;

	protected int minSize;

	protected int maxSize;
	
	protected AbstractClassifier newMemberModel;

	public GeneticLayer(Instances instances, int maxNumBerOfGenerations, boolean implicitReset, Type[] types, Random rand, int minSize, int maxSize, AbstractClassifier newMemberModel) {
		this.instances = instances;
		this.maxNumBerOfGenerations = maxNumBerOfGenerations;
		this.implicitReset = implicitReset;
		this.rand = rand;
		this.startTypes = types;
		this.minSize = minSize;
		this.maxSize = maxSize;
		this.newMemberModel = newMemberModel;
	}

	public Type[] types() {
		return finalPopulation;
	}

	public void train(Instances instances) {
		for (Type type : finalPopulation) {
			type.train(instances);
		}
	}

	public void run() {
		List<Solve> solves = new ArrayList<>();
		for (int i = 0; i < startTypes.length; i++) {
			solves.add(new Solve(startTypes[i], startTypes[i].classify(instances)));
		}

		for (int i = 0; i < maxNumBerOfGenerations; i++) {
			List<Solve> selection = selection(solves);
			List<Solve> childs = new ArrayList<>();
			for (int j = 0; j < selection.size()-1; j += 2) {
				Solve[] c = crossover(selection.get(j), selection.get(j + 1));
				mutation(c[0]);
				mutation(c[1]);
				childs.add(c[0]);
				childs.add(c[1]);
			}
			solves = childs;
		}

		finalPopulation = new Type[solves.size()];
		for (int i = 0; i < solves.size(); i++) {
			finalPopulation[i] = solves.get(i).type;
		}
	}

	public List<Solve> selection(List<Solve> solves) {
		double avg = 0;
		for (Solve solve : solves) {
			avg += solve.cost;
		}
		avg = avg / (double) solves.size();

		List<Solve> aboveAvg = new ArrayList<>();
		for (Solve solve : solves) {
			if (solve.cost >= avg) {
				aboveAvg.add(solve);
			}
		}
		Collections.sort(aboveAvg);
		return aboveAvg;
	}

	public Solve[] crossover(Solve a, Solve b) {
		// 2 points crossover

		int size = instances.numAttributes() - 1;

		int index1 = rand.nextInt(size / 2);
		int index2 = size / 2 + rand.nextInt(size/2);

		boolean[] c = a.type.attributes.clone();
		boolean[] d = b.type.attributes.clone();
		for (int i = index1; i < index2; i++) {
			c[i] = b.type.attributes[i];
			d[i] = a.type.attributes[i];
		}
		Type cs = new Type(c, minSize, maxSize, newMemberModel);
		Type ds = new Type(d, minSize, maxSize, newMemberModel);
		return new Solve[] { new Solve(cs, cs.classify(instances)), new Solve(ds, ds.classify(instances)) };
	}

	public void mutation(Solve solve) {
		if (rand.nextDouble() < 0.05) {
			int numAttributes = solve.type.attributes.length;
			int index = rand.nextInt(numAttributes);
			solve.type.attributes[index] = !solve.type.attributes[index];
		}
	}

	private static class Solve implements Comparable<Solve> {

		public double cost;

		public Type type;

		public Solve(Type type, double cost) {
			this.type = type;
			this.cost = cost;
		}

		@Override
		public int compareTo(Solve b) {
			return cost > b.cost ? 1 : (cost == b.cost ? 0 : -1);
		}
	}
}
