package moa.classifiers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.ListOption;
import com.github.javacliparser.Option;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.eocd.EnsembleMemberConfiguration;
import moa.classifiers.eocd.GeneticAlgorithm;
import moa.classifiers.eocd.MemberClassifier;
import moa.classifiers.eocd.Problem;
import moa.classifiers.eocd.Solve;
import moa.classifiers.eocd.StopCondition;
import moa.classifiers.functions.Perceptron;
import moa.classifiers.lazy.kNN;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.Utils;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

public class EnsembleOptimizationForConceptDrift extends AbstractClassifier implements MultiClassClassifier {

	private static final long serialVersionUID = 1L;

	public FloatOption betaOption = new FloatOption("beta", 'b', "Factor to punish mistakes by.", 0.95, 0.0, 1.0);

	public IntOption initialMembersSize = new IntOption("initialEnsembleSize", 's', "Initial quantity of ensemble members.", 10, 1, Integer.MAX_VALUE);

	public IntOption hiddenMemberSize = new IntOption("hiddenMemberSize", 'h', "Quantity of hidden members.", 10, 0, Integer.MAX_VALUE);

	public IntOption maxMembersSize = new IntOption("maxMembersSize", 'm', "Maximum quantity of ensemble members.", 50, 1, Integer.MAX_VALUE);

	public IntOption optimizationFrequency = new IntOption("optimizationFrequency", 'o', "Number of Instances Between Optimization Process.", 1000, 1,
			Integer.MAX_VALUE);

	public FlagOption sampleConfigurationBeforeOptimization = new FlagOption("sampleConfigurationBeforeOptimization", 'B',
			"Use of atribute selection for member classifiers.");

	public FlagOption sampleConfigurationAfterOptimization = new FlagOption("sampleConfigurationAfterOptimization", 'A',
			"Use of atribute selection for member classifiers.");

	public IntOption epochs = new IntOption("epochs", 'e', "Number of iterations of the Optimization Algorithm", 100, 1, Integer.MAX_VALUE);
	
	public FlagOption useAttributeSelection = new FlagOption("useAttributeSelection", 'a', "Use of atribute selection for member classifiers.");

	public FlagOption useOptimization = new FlagOption("useOptimization", 'O', "Use of atribute selection for member classifiers.");

	public FlagOption useOptimizationFrequency = new FlagOption("useOptimizationFrequency", 'Y', "Use of atribute selection for member classifiers.");
	
	public FlagOption resetBufferOnOptimization = new FlagOption("resetBufferOnOptimization", 'R', "Use of atribute selection for member classifiers.");

	public FlagOption useThread = new FlagOption("useThread", 'T', "Use of atribute selection for member classifiers.");
	
	public IntOption limitOption = new IntOption("limit", 'w', "The maximum number of instances to storeo cost evaluation", 1000, 1, Integer.MAX_VALUE);

	public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd', "Drift detection method to use.", ChangeDetector.class,
			"RDDM");

	public ListOption learnerListOption = new ListOption("learners", 'l', "The learners to combine.",
			new ClassOption("learner", ' ', "", Classifier.class, "trees.HoeffdingTree"),
			new Option[] { new ClassOption("", ' ', "", Classifier.class, "trees.HoeffdingTree -l MC"),
					new ClassOption("", ' ', "", Classifier.class, "trees.HoeffdingTree -l NB"),
					new ClassOption("", ' ', "", Classifier.class, "trees.HoeffdingTree -l NBAdaptive"),
					new ClassOption("", ' ', "", Classifier.class, "bayes.NaiveBayes"), // 3 for uniform probability
					new ClassOption("", ' ', "", Classifier.class, "bayes.NaiveBayes"), new ClassOption("", ' ', "", Classifier.class, "bayes.NaiveBayes"),
					new ClassOption("", ' ', "", Classifier.class, "functions.Perceptron -r 0.1"),
					new ClassOption("", ' ', "", Classifier.class, "functions.Perceptron -r 0.01"),
					new ClassOption("", ' ', "", Classifier.class, "functions.Perceptron -r 0.001") },
			',');

	protected ChangeDetector driftDetectionMethod;

	// initial members has all attributes
	int numAttributes = -1; // -1 for unknown

	public MemberClassifier[] classifiers;
	public Instances buffer;

	private Classifier[] possibleMembers;
	private Thread optimizationThread;
	public GeneticAlgorithm optimizationAlgorithm;
	public List<EnsembleMemberConfiguration> configurations = new ArrayList<>();

	private enum DriftLevel {
		INCONTROL, OUTCONTROL, WARNING
	};

	private DriftLevel driftState = DriftLevel.INCONTROL;

	@Override
	public String getPurposeString() {
		return "Optimization Choice Ensemble For Data Streams With Concept Drift.";
	}

	@Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		super.prepareForUseImpl(monitor, repository);

		driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
		classifiers = new MemberClassifier[maxMembersSize.getValue()];

		initBaseLearners(monitor, repository);
		initMembers();
	}

	private void initBaseLearners(TaskMonitor monitor, ObjectRepository repository) {
		Option[] learnerOptions = this.learnerListOption.getList();
		possibleMembers = new Classifier[learnerOptions.length];
		for (int i = 0; i < learnerOptions.length; i++) {
			possibleMembers[i] = (AbstractClassifier) ((ClassOption) learnerOptions[i]).materializeObject(monitor, repository);
			if (monitor.taskShouldAbort()) {
				return;
			}
			possibleMembers[i].prepareForUse(monitor, repository);
			if (monitor.taskShouldAbort()) {
				return;
			}
		}
	}

	private void initMembers() {
		for (int i = 0; i < this.classifiers.length; i++) {
			if (classifiers[i] == null) {
				classifiers[i] = createMember();
			}
			classifiers[i].weight = 1.0;
			if (i < initialMembersSize.getValue()) {
				classifiers[i].isActive = true;
			} else if (i - initialMembersSize.getValue() < hiddenMemberSize.getValue()) {
				classifiers[i].isHidden = true;
			}
		}
	}

	protected MemberClassifier createMember() {
		int index = this.classifierRandom.nextInt(possibleMembers.length);
		AbstractClassifier classifier = (AbstractClassifier) possibleMembers[index].copy();
		classifier.resetLearning();
		MemberClassifier member = new MemberClassifier(classifier, index, classifierRandom, numAttributes);
		return member;
	}

	@Override
	public void setModelContext(InstancesHeader ih) {
		super.setModelContext(ih);
		if (useAttributeSelection.isSet()) {
			numAttributes = ih.numAttributes();
			classifiers = new MemberClassifier[maxMembersSize.getValue()];
			initMembers();
		}
		try {
			buffer = new Instances(ih, 0);
			buffer.setClassIndex(ih.classIndex());
		} catch (Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	@Override
	@SuppressWarnings("deprecation")
	public void resetLearningImpl() {
		buffer = null;
		numAttributes = -1;
		if (driftDetectionMethod != null) {
			driftDetectionMethod.resetLearning();
		}
		if (trainingHasStarted()) {
			initMembers();
		}
		if (optimizationThread != null) {
			optimizationThread.stop();
		}
		optimizationThread = null;
	}

	int countInstances = 0;

	@Override
	public void trainOnInstanceImpl(Instance inst) {

		if (useOptimizationFrequency.isSet() && (countInstances++ % optimizationFrequency.getValue() == 0)) {
			driftDetected();
		}

		// drift detection
		int y = Utils.maxIndex(getVotesForInstance(inst));
		int real = (int) inst.classValue();
		driftObserver(y, real);

		// update weights
		double instWeight = inst.weight();
		double totalWeight = 0.0;
		boolean doupdate = false;
		double beta = betaOption.getValue();
		for (int i = 0; i < classifiers.length; i++) {
			MemberClassifier member = classifiers[i];
			if (member.isActive) {
				if (!member.correctlyClassifies(inst)) {
					member.weight = member.weight * beta * instWeight;
					doupdate = true;
					if (member.weight < 0.001) {
						classifiers[i] = createMember();
						classifiers[i].isHidden = true;
					}
				}
				if (member.isActive) {
					totalWeight += member.weight;
				}
			}
		}

		// normalize weights
		for (int i = 0; i < classifiers.length; i++) {
			MemberClassifier member = classifiers[i];
			if (doupdate && member.isActive) {
				member.weight = member.weight / totalWeight;
			}
			if (member.isActive || member.isHidden) {
				member.train(inst);
			}
		}

		// update buffer
		if (buffer == null) {
			buffer = new Instances(inst.dataset());
		}
		buffer.add(inst);
		if (buffer.numInstances() >= limitOption.getValue()) {
			buffer.delete(0);
		}
	}

	protected void driftObserver(int y, int real) {
		DriftLevel newLevel = DriftLevel.INCONTROL;

		driftDetectionMethod.input(y == real ? 0.0 : 1.0);
		if (this.driftDetectionMethod.getChange()) {
			newLevel = DriftLevel.OUTCONTROL;
		} else if (this.driftDetectionMethod.getWarningZone()) {
			newLevel = DriftLevel.WARNING;
		}
		if (newLevel != driftState) {
			switch (newLevel) {
			case OUTCONTROL:
				driftDetected();
				driftState = DriftLevel.OUTCONTROL;
				break;
			case WARNING:
				driftWarning();
				driftState = DriftLevel.WARNING;
				break;
			case INCONTROL:
				driftState = DriftLevel.INCONTROL;
				break;
			}
		}
	}

	protected void driftWarning() {

	}

	protected boolean isRunningOptimization() {
		return optimizationThread != null;
	}

	protected void driftDetected() {
		driftDetectedWhitTread();
	}

	protected void driftDetectedWhitoutThread() {
		if (!useOptimization.isSet()) {
			return;
		}

		if (isRunningOptimization()) { // running optimization
			return;
		}

		// System.out.println("Drift at Instance " + countInstances);
		// System.out.println("Drift: State Before Optimization");
		// printSummary();

		if (sampleConfigurationBeforeOptimization.isSet()) {
			configurations.add(new EnsembleMemberConfiguration(classifiers));
		}

		GeneticAlgorithm optimizer = makeOptimizer();
		if (resetBufferOnOptimization.isSet()) {
			buffer = null;
		}

		List<Integer> active = new ArrayList<>();
		for (int i = 0; i < classifiers.length; i++) {
			if (classifiers[i].isActive) {
				active.add(i);
			}
		}
		System.out.println("Active Classifiers Before:" + active);

		optimizer.execute();
		Solve bestSolve = (Solve) optimizer.bestSolve();
		double[] weights = optimizer.problem.getWeights(bestSolve);

		active.clear();

		int numHidden = 0;
		for (int i = 0; i < classifiers.length; i++) {
			classifiers[i].weight = weights[i];
			classifiers[i].isActive = weights[i] > 0;
			classifiers[i].isHidden = false;
			if (!classifiers[i].isActive && numHidden < hiddenMemberSize.getValue()) {
				classifiers[i] = createMember();
				classifiers[i].isHidden = true;
				numHidden += 1;
			}
			if (classifiers[i].isActive) {
				active.add(i);
			}
		}
		System.out.println("Active Classifiers After:" + active);

		if (sampleConfigurationAfterOptimization.isSet()) {
			configurations.add(new EnsembleMemberConfiguration(classifiers));
		}

		optimizationThread = null;

		// System.out.println("Drift: State After Optimization");
		// printSummary();
		// System.out.println("-------------------------------------");

	}
	
	private GeneticAlgorithm makeOptimizer() {
		Problem problem = new Problem(classifiers, buffer, configurations);
		StopCondition stopCondition = new StopCondition(0, 2000, 0, epochs.getValue(), epochs.getValue(), 0, 120);
		Random rand = new Random(super.randomSeed);
		GeneticAlgorithm optimizer = new GeneticAlgorithm(problem, stopCondition, rand);
		return optimizer;
	}

	protected void driftDetectedWhitTread() {
		// with thread
		if (!useOptimization.isSet()) {
			return;
		}

		if (isRunningOptimization()) { // running optimization
			return;
		}

		// System.out.println("Drift at Instance " + countInstances);
		// System.out.println("Drift: State Before Optimization");
		// printSummary();

		if (sampleConfigurationBeforeOptimization.isSet()) {
			configurations.add(new EnsembleMemberConfiguration(classifiers));
		}

		
		GeneticAlgorithm optimizer = makeOptimizer();
		if (resetBufferOnOptimization.isSet()) {
			buffer = null;
		}

		List<Integer> active = new ArrayList<>();
		for (int i = 0; i < classifiers.length; i++) {
			if (classifiers[i].isActive) {
				active.add(i);
			}
		}

		optimizationThread = new Thread() {
			public void run() {
				try {
					optimizer.execute();

					Solve bestSolve = (Solve) optimizer.bestSolve();
					double[] weights = optimizer.problem.getWeights(bestSolve);

					List<Integer> active = new ArrayList<>();

					int numHidden = 0;
					for (int i = 0; i < classifiers.length; i++) {
						classifiers[i].weight = weights[i];
						classifiers[i].isActive = weights[i] > 0;
						classifiers[i].isHidden = false;
						if (!classifiers[i].isActive && numHidden < hiddenMemberSize.getValue()) {
							classifiers[i] = createMember();
							classifiers[i].isHidden = true;
							numHidden += 1;
						}
						if (classifiers[i].isActive) {
							active.add(i);
						}
					}

					if (sampleConfigurationAfterOptimization.isSet()) {
						configurations.add(new EnsembleMemberConfiguration(classifiers));
					}

				} catch (Error e) {
					System.err.println("Optmizaition error!!!");
				}

				optimizationThread = null;

				// System.out.println("Drift: State After Optimization");
				// printSummary();
				// System.out.println("-------------------------------------");

			}
		};
		optimizationThread.start();
		
		if(!useThread.isSet()) {
			try {
				optimizationThread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}	
		}

	}

	public void printSummary() {

		Map<Integer, Integer> hashToIndex = new HashMap<>();
		hashToIndex.put(kNN.class.getSimpleName().hashCode(), 0);
		hashToIndex.put(HoeffdingTree.class.getSimpleName().hashCode(), 1);
		hashToIndex.put(Perceptron.class.getSimpleName().hashCode(), 2);
		hashToIndex.put(NaiveBayes.class.getSimpleName().hashCode(), 3);

		int[] memberCount = new int[4];

		int count = 0;
		StringBuilder str = new StringBuilder(1000);
		for (MemberClassifier member : classifiers) {
			if (member.isActive) {
				memberCount[hashToIndex.get(member.type)] += 1;
				str.append(", ");
				str.append(member);
				count += 1;
			}
		}
		System.out.println(count + " " + Arrays.toString(memberCount) + " " + str.toString());
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		DoubleVector combinedVote = new DoubleVector();
		if (this.trainingWeightSeenByModel > 0.0) {
			for (MemberClassifier member : this.classifiers) {
				if (member.isActive) {
					DoubleVector vote = new DoubleVector(member.distribution(inst));
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						vote.scaleValues(member.weight);
						combinedVote.addValues(vote);
					}
				}
			}
		}
		return combinedVote.getArrayRef();
	}

	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		Measurement[] measurements = null;
		if (this.classifiers != null) {
			measurements = new Measurement[this.classifiers.length];
			for (int i = 0; i < this.classifiers.length; i++) {
				measurements[i] = new Measurement("member weight " + (i + 1), this.classifiers[i].weight);
			}
		}
		return measurements;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		out.append(this.getPurposeString());
	}
}
