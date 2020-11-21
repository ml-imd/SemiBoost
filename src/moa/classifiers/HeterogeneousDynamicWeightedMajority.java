package moa.classifiers;


import java.util.ArrayList;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.ListOption;
import com.github.javacliparser.Option;
import com.yahoo.labs.samoa.instances.Instance;

import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;
import weka.core.Utils;

//Dynamic weighted majority algorithm. Heterogeneos mod
public class HeterogeneousDynamicWeightedMajority extends AbstractClassifier implements MultiClassClassifier {

	private static final long serialVersionUID = 1L;

	public ListOption learnerListOption = new ListOption("learners", 'l', "The learners to combine.",
			new ClassOption("learner", ' ', "", Classifier.class, "trees.HoeffdingTree"),
			new Option[] { new ClassOption("", ' ', "", Classifier.class, "trees.HoeffdingTree -l MC"),
					new ClassOption("", ' ', "", Classifier.class, "trees.HoeffdingTree -l NB"),
					new ClassOption("", ' ', "", Classifier.class, "trees.HoeffdingTree -l NBAdaptive"),
					new ClassOption("", ' ', "", Classifier.class, "bayes.NaiveBayes"), // 3 for uniform probability
					new ClassOption("", ' ', "", Classifier.class, "bayes.NaiveBayes"),
					new ClassOption("", ' ', "", Classifier.class, "bayes.NaiveBayes"),
					new ClassOption("", ' ', "", Classifier.class, "functions.Perceptron -r 0.1"),
					new ClassOption("", ' ', "", Classifier.class, "functions.Perceptron -r 0.01"),
					new ClassOption("", ' ', "", Classifier.class, "functions.Perceptron -r 0.001") },
			',');

	public IntOption periodOption = new IntOption("period", 'p',
			"Period between expert removal, creation, and weight update.", 50, 1, Integer.MAX_VALUE);

	public FloatOption betaOption = new FloatOption("beta", 'b', "Factor to punish mistakes by.", 0.5, 0.0, 1.0);

	public FloatOption thetaOption = new FloatOption("theta", 't', "Minimum fraction of weight per model.", 0.01, 0.0,
			1.0);

	public IntOption maxExpertsOption = new IntOption("maxExperts", 'e', "Maximum number of allowed experts.",
			Integer.MAX_VALUE, 2, Integer.MAX_VALUE);

	protected List<Classifier> experts;
	protected List<Double> weights;
	protected long epochs;

	private Classifier[] possibleMembers;

	@Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		Option[] learnerOptions = this.learnerListOption.getList();
		possibleMembers = new Classifier[learnerOptions.length];
		for (int i = 0; i < learnerOptions.length; i++) {
			possibleMembers[i] = (AbstractClassifier) ((ClassOption) learnerOptions[i]).materializeObject(monitor,
					repository);
			if (monitor.taskShouldAbort()) {
				return;
			}
			possibleMembers[i].prepareForUse(monitor, repository);
			if (monitor.taskShouldAbort()) {
				return;
			}
		}
		super.prepareForUseImpl(monitor, repository);
	}

	protected Classifier createMember() {
		int index = this.classifierRandom.nextInt(possibleMembers.length);
		Classifier classifier = possibleMembers[index].copy();
		return classifier;
	}

	@Override
	public void resetLearningImpl() {
		this.experts = new ArrayList<>(50);
		Classifier classifier = createMember();
		classifier.resetLearning();
		this.experts.add(classifier);
		this.weights = new ArrayList<>(50);
		this.weights.add(1.0);
		this.epochs = 0;
	}

	protected void scaleWeights(double maxWeight) {
		double sf = 1.0 / maxWeight;
		for (int i = 0; i < weights.size(); i++) {
			weights.set(i, weights.get(i) * sf);
		}
	}

	protected void removeExperts() {
		for (int i = experts.size() - 1; i >= 0; i--) {
			if (weights.get(i) < this.thetaOption.getValue()) {
				experts.remove(i);
				weights.remove(i);
			} // if
		} // for
	} // DWM::removeExperts

	protected void removeWeakestExpert(int i) {
		experts.remove(i);
		weights.remove(i);
	} // DWM::removeWeakestExpert

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		this.epochs++;
		double[] Pr = new double[inst.numClasses()];
		double maxWeight = 0.0;
		double weakestExpertWeight = 1.0;
		int weakestExpertIndex = -1;
		// Loop over experts
		for (int i = 0; i < this.experts.size(); i++) {
			double[] pr = this.experts.get(i).getVotesForInstance(inst);
			int yHat = Utils.maxIndex(pr);
			if ((yHat != (int) inst.classValue()) && this.epochs % this.periodOption.getValue() == 0) {
				this.weights.set(i, this.weights.get(i) * this.betaOption.getValue());
			}
			Pr[yHat] += this.weights.get(i);
			maxWeight = Math.max(maxWeight, this.weights.get(i));
			if (this.weights.get(i) < weakestExpertWeight) {
				weakestExpertIndex = i;
				weakestExpertWeight = weights.get(i);
			}
		}
		int yHat = Utils.maxIndex(Pr);
		if (this.epochs % this.periodOption.getValue() == 0) {
			scaleWeights(maxWeight);
			removeExperts();
			if (yHat != (int) inst.classValue()) {
				if (experts.size() == this.maxExpertsOption.getValue()) {
					removeWeakestExpert(weakestExpertIndex);
				}
				Classifier classifier = createMember();
				classifier.resetLearning();
				this.experts.add(classifier);
				this.weights.add(1.0);
			}
		}
		// train experts
		for (Classifier expert : this.experts) {
			expert.trainOnInstance(inst);
		}
	}

	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		double[] Pr = new double[inst.numClasses()];
		for (int i = 0; i < this.experts.size(); i++) {
			double[] pr = this.experts.get(i).getVotesForInstance(inst);
			int yHat = Utils.maxIndex(pr);
			Pr[yHat] += this.weights.get(i);
		} // for
		Utils.normalize(Pr);
		return Pr;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		Measurement[] measurements = null;
		if (this.weights != null) {
			measurements = new Measurement[1];
			measurements[0] = new Measurement("members size", this.weights.size());
		}
		return measurements;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {

	}
}
