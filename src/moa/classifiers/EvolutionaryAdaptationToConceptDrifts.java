package moa.classifiers;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.eacd.GeneticLayer;
import moa.classifiers.eacd.ReplicatorDynamics;
import moa.classifiers.eacd.Type;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.Utils;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

public class EvolutionaryAdaptationToConceptDrifts extends AbstractClassifier implements MultiClassClassifier {

	private static final long serialVersionUID = 1L;

	public IntOption blockSize = new IntOption("blockSize", 'b', "", 100, 1, Integer.MAX_VALUE);

	public IntOption numInitialMembers = new IntOption("numInitialMembers", 'I', "", 30, 1, Integer.MAX_VALUE);

	public IntOption maxMemberByType = new IntOption("maxMemberByType", 'M', "", 20, 1, Integer.MAX_VALUE);

	public IntOption minMemberByType = new IntOption("minMemberByType", 'm', "", 1, 1, Integer.MAX_VALUE);

	public IntOption limitOption = new IntOption("limit", 'w', "", 100, 1, Integer.MAX_VALUE);

	public IntOption epochs = new IntOption("epochs", 'e', "", 15, 1, Integer.MAX_VALUE);

	public FloatOption rateAttOption = new FloatOption("rateAttOption", 'p', "", 0.6, 0.0, 1.0);

	public FlagOption useImplicitReset = new FlagOption("implicitReset", 'r', "");

	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

	public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd', "Drift detection method to use.", ChangeDetector.class, "EDDM");

	protected ChangeDetector driftDetectionMethod;

	private enum DriftLevel {
		INCONTROL, OUTCONTROL, WARNING
	};

	private DriftLevel driftState;

	private Instances buffer;

	private Instances block;

	private ReplicatorDynamics baseLayer;

	private GeneticLayer geneticLayer;

	private int numAttributes;

	private double ensembleAccuracy;

	private boolean geneticLayerMature;

	private AbstractClassifier modelbase;

	private LinkedList<Double> baseAcc;
	private LinkedList<Double> genAcc;

	@Override
	public String getPurposeString() {
		return "Optimization Choice Ensemble For Data Streams With Concept Drift.";
	}

	@Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		super.prepareForUseImpl(monitor, repository);

		modelbase = (AbstractClassifier) getPreparedClassOption(this.baseLearnerOption);
		driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
	}

	@Override
	public void resetLearningImpl() {
		baseAcc = new LinkedList<>();
		genAcc = new LinkedList<>();
		geneticLayerMature = false;
		ensembleAccuracy = 1.0;
		driftState = DriftLevel.INCONTROL;
		numAttributes = -1;
		if (driftDetectionMethod != null) {
			driftDetectionMethod.resetLearning();
		}
	}

	@Override
	public void setModelContext(InstancesHeader ih) {
		super.setModelContext(ih);
		this.numAttributes = ih.numAttributes();
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
	public void trainOnInstanceImpl(Instance inst) {
		if (baseLayer == null) {
			baseLayer = new ReplicatorDynamics(numAttributes, numInitialMembers.getValue(), rateAttOption.getValue(), super.classifierRandom, minMemberByType.getValue(), maxMemberByType.getValue(), modelbase);
		}

		if (!useImplicitReset.isSet() && isGeneticLayerActive()) {
			// drift detection
			int y = Utils.maxIndex(getVotesForInstance(inst));
			int real = (int) inst.classValue();
			driftObserver(y, real);
		}

		// update buffer
		if (buffer == null) {
			buffer = new Instances(inst.dataset());
		}
		buffer.add(inst);
		if (buffer.numInstances() >= limitOption.getValue()) {
			buffer.delete(0);
		}

		if (block == null) {
			block = new Instances(inst.dataset());
		}
		block.add(inst);
		if (block.numInstances() >= blockSize.getValue()) {
			trainBlock(block);
			block = null;
		}
	}

	protected void trainBlock(Instances instances) {
		baseLayer.train(instances);
		if (baseLayer.isMature()) {
			if (geneticLayer == null) {
				driftDetected(); // reset GA
			}
			geneticLayer.train(instances);

			double acc = classify(instances, geneticLayer.types());
			geneticLayerMature = acc > ensembleAccuracy;

			baseAcc.offer(classify(instances, baseLayer.types()));
			genAcc.offer(classify(instances, geneticLayer.types()));
			if (baseAcc.size() > 10) {
				baseAcc.poll();
				genAcc.poll();
			}
		}

		ensembleAccuracy = (classifyEnsemble(instances) + ensembleAccuracy) / 2.0;

		if (useImplicitReset.isSet()) {
			if (baseAcc.size() == 10) { // last 10 data blocks
				double bacc = 0;
				double gacc = 0;
				for (double d : baseAcc) {
					bacc += d;
				}
				for (double d : genAcc) {
					gacc += d;
				}
				if (gacc <= bacc) {
					driftDetected();
				}
			}
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

	protected void driftDetected() {
		geneticLayer = new GeneticLayer(buffer, epochs.getValue(), useImplicitReset.isSet(), baseLayer.types(), super.classifierRandom, minMemberByType.getValue(), maxMemberByType.getValue(), modelbase);
		geneticLayer.run();
	}

	protected boolean isGeneticLayerActive() {
		return geneticLayer != null && geneticLayerMature;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		DoubleVector combinedVote = new DoubleVector();
		if (this.trainingWeightSeenByModel > 0.0) {
			for (Type type : baseLayer.types()) {
				for (double[] memberVote : type.distribution(inst)) {
					DoubleVector vote = new DoubleVector(memberVote);
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						combinedVote.addValues(vote);
					}
				}
			}
			if (isGeneticLayerActive()) {
				for (Type type : geneticLayer.types()) {
					for (double[] memberVote : type.distribution(inst)) {
						DoubleVector vote = new DoubleVector(memberVote);
						if (vote.sumOfValues() > 0.0) {
							vote.normalize();
							vote.scaleValues(2);
							combinedVote.addValues(vote);
						}
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
	public void getModelDescription(StringBuilder out, int indent) {
		out.append(this.getPurposeString());
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		List<Measurement> measurements = new ArrayList<>();
		for (int i = 0; i < baseLayer.types().length; i++) {
			new Measurement("member weight " + (measurements.size() + 1), 1);
		}
		if (isGeneticLayerActive()) {
			for (int i = 0; i < geneticLayer.types().length; i++) {
				new Measurement("member weight " + (measurements.size() + 1), 2);
			}
		}
		if (measurements.isEmpty()) {
			return null;
		} else {
			return measurements.toArray(new Measurement[] {});
		}
	}

	private double classifyEnsemble(Instances instances) {
		double correct = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance instance = instances.get(i);
			int y = Utils.maxIndex(getVotesForInstance(instance));
			if (y == (int) instance.classValue()) {
				correct += 1;
			}
		}
		return correct / (double) instances.numInstances();
	}

	private static double classify(Instances instances, Type[] types) {
		double correct = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance instance = instances.get(i);
			int y = Utils.maxIndex(distribution(instance, types));
			if (y == (int) instance.classValue()) {
				correct += 1;
			}
		}
		return correct / (double) instances.numInstances();
	}

	private static double[] distribution(Instance instance, Type[] types) {
		double[] votes = new double[instance.numClasses()];
		for (Type type : types) {
			double[][] dist = type.distribution(instance);
			for (double[] v : dist) {
				int y = Utils.maxIndex(v);
				votes[y] += 1;
			}
		}
		return votes;
	}
}
