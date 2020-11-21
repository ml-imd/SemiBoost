package moa.classifiers.eocd;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.functions.Perceptron;
import moa.classifiers.functions.SGD;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Utils;

public class Problem {
	
	public final int sizeEncode;

	public final int numObjectives;

	public int numMaxInitialSolves = 30;

	public int numMaxConfigurations = 30;

	private Instances instances;

	public MemberClassifier[] classifiers;
	
	private Set<Integer> activeClassifiers;
	
	private int [] solveToIndex;

	public List<EnsembleMemberConfiguration> configurations;

	// each member has two values: (active : boolean, weight : real)
	public Problem(MemberClassifier[] classifiers, Instances instances, List<EnsembleMemberConfiguration> configurations) {
		this.sizeEncode = numActiveClassifiers(classifiers);
		this.numObjectives = 1;
		
		this.configurations = configurations;
		this.classifiers = classifiers;
		this.instances = instances;
		initClassifiers();
	}
	
	public String createRepresentation (Solve solve) {
		Map<Integer, Integer> hashToIndex = new HashMap<>();
		hashToIndex.put(HoeffdingTree.class.getSimpleName().hashCode(), 0);
		hashToIndex.put(Perceptron.class.getSimpleName().hashCode(), 1);
		hashToIndex.put(NaiveBayes.class.getSimpleName().hashCode(), 2);
		hashToIndex.put(SGD.class.getSimpleName().hashCode(), 3);

		int[] memberCount = new int[hashToIndex.size()];
		Set <Integer> active = activeClassifiers();
		for (int i = 0; i < active.size(); i++) {
			if(solve.isActive(i)) {
				memberCount[hashToIndex.get(classifiers[i].type)] += 1;
			}
		}
		return String.format("(%.4f, %s)", solve.cost(), Arrays.toString(memberCount));
	}
	
	private static int numActiveClassifiers (MemberClassifier[] classifiers) {
		int count = 0;
		for(MemberClassifier member : classifiers) {
			if (member.isActive || member.isHidden) {
				count += 1;
			}
		}
		return count;
	}
	
	private void initClassifiers () {
		activeClassifiers = new HashSet<>();
		solveToIndex = new int [numActiveClassifiers(classifiers)];
		for(int i=0;i<classifiers.length;i++) {
			MemberClassifier member = classifiers[i];
			if (member.isActive || member.isHidden) {
				solveToIndex[activeClassifiers.size()] = i;
				activeClassifiers.add(i);
			}
		}
	}
	
	public Set<Integer> activeClassifiers() {
		return activeClassifiers;
	}

	public void evaluate(Solve solve) {
		double[] weights = getWeights(solve);
		double error = 0;
		try {
			for (int i = 0; i < instances.numInstances(); i++) {
				Instance inst = instances.get(i);
				int y = Utils.maxIndex(getVotesForInstance(inst, weights));
				int real = (int) inst.classValue();
				if (y != real) {
					error += 1;
				}
			}
			error = error / instances.numInstances();
		} catch (Exception e) {
			error = 1.0;
			System.err.println("Evaluation ERROR!!!" + solve);
		}
		solve.cost(error);
	}

	public double[] getVotesForInstance(Instance inst, double[] weights) {
		// use only active classifiers
		DoubleVector combinedVote = new DoubleVector();
		Set <Integer> active = activeClassifiers();
		for (int i = 0; i < active.size(); i++) {
			if (weights[i] > 0) {
				DoubleVector vote = new DoubleVector(classifiers[solveToIndex[i]].distribution(inst));
				if (vote.sumOfValues() > 0.0) {
					vote.normalize();
					vote.scaleValues(weights[i]);
					combinedVote.addValues(vote);
				}
			}
		}
		return combinedVote.getArrayRef();
	}
	
	public void setWeight (Solve _solve, int member, double weight) {
		Solve solve = (Solve) _solve;
		int index = 0;
		for(int i=0;i<solveToIndex.length;i++) {
			if(solveToIndex[i] == member) {
				index = i;
				break;
			}
		}
		solve.weight(index, weight);
	}

	public double[] getWeights(Solve _solve) {
		Solve solve = (Solve) _solve;
		double[] weights = new double[classifiers.length];
		Set <Integer> active = activeClassifiers();
		for (int i = 0; i < active.size(); i++) {
			weights[solveToIndex[i]] = solve.weight(i);
		}
		return weights;
	}

	public Solve createEmptySolve() {
		return new Solve(this);
	}
}
