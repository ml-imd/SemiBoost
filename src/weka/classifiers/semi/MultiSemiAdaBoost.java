package weka.classifiers.semi;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.converters.ArffLoader;

public class MultiSemiAdaBoost extends RandomizableIteratedSingleClassifierEnhancer implements OptionHandler, TechnicalInformationHandler, AdditionalMeasureProducer {

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain command line options (see setOptions)
	 * @throws Exception 
	 */
	public static void main(String[] argv) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File("iris.arff"));
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		Instances test = new Instances(instances, 0);
		Instances train = new Instances(instances, 0);
		for (int i = 0; i < 15; i++) {
			test.add(instances.instance(i));
			test.add(instances.instance(i + 50));
			test.add(instances.instance(i + 100));
		}
		for (int i = 15; i < 50; i++) {
			train.add(instances.instance(i));
			train.add(instances.instance(i + 50));
			train.add(instances.instance(i + 100));
		}
		for (int i = 15; i < train.numInstances(); i++) {
			train.instance(i).setClassMissing();
		}

		double acc = 0;
		for (int i = 0; i < 30; i++) {
			MultiSemiAdaBoost classifier = new MultiSemiAdaBoost();
			classifier.setSeed(i);
			classifier.buildClassifier(new Instances(train));
			Evaluation eval = new Evaluation(new Instances(train));
			eval.evaluateModel(classifier, test);
			acc += eval.pctCorrect();
			System.out.println(eval.pctCorrect()); //TODO print
		}
		System.out.println("AVG " + acc / 30.0);

		//runClassifier(new SemiBoost(), argv);
	}

	private static final long serialVersionUID = 1;

	protected double constant1 = -1; // set default #l/#u as -1
	protected double constant2 = -1;
	protected double constant3 = -1;
	protected double percentSampling = 0.15;
	protected double deltaPercentile = 0.1; // 10 a 20
	
	private List<WeightedClassifier> classifiers;
	private double delta;
	private Instances labeled;

	public MultiSemiAdaBoost() {
		setClassifier(new SMO());
		setNumIterations(20);
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if an error occurred during the prediction
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] output = new double[instance.numClasses()];
		if (classifiers.isEmpty()) {
			return output;
		}

		for (WeightedClassifier c : classifiers) {
			double[] y = c.distributionForInstance(instance);
			for (int i = 0; i < output.length; i++) {
				output[i] += y[i] * c.weight;
			}
		}

		double sum = 0;
		for (int i = 0; i < output.length; i++) {
			sum += output[i];
		}
		for (int i = 0; i < output.length; i++) {
			output[i] = output[i] / sum;
		}
		return output;
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double[] output = distributionForInstance(instance);
		int max = 0;
		for (int i = 1; i < output.length; i++) {
			if (output[i] > output[max]) {
				max = i;
			}
		}
		return max;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances set of instances serving as training data 
	 * @throws Exception if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		if (getClassifier() == null) {
			throw new Exception("A base classifier has not been specified!");
		}

		Set<Instance> labeled = new HashSet<>();
		Set<Instance> unlabeled = new HashSet<>();
		splitInstances(instances, labeled, unlabeled);

		// default or 1
		if(constant1 == -1)
			constant1 = 1.0 / (double) labeled.size();
		if(constant2 == -1)
			constant2 = 1.0 / (double) instances.numInstances();
		if(constant3 == -1)
			constant3 = unlabeled.size() > 0 ? 1.0 / (double) (2.0 * unlabeled.size()) : 0;

		classifiers = new ArrayList<>(getNumIterations());
		Map<InstancePair, Double> similarities = computeSimilarity(instances);

		Random rand = new Random(getSeed());
		do {
			Instances sampled = new Instances(this.labeled);
			Map<Instance, Confidence> confidence = sampling(sampled, instances, similarities, labeled, unlabeled, rand);
			WeightedClassifier classifier = new WeightedClassifier();
			classifier.buildClassifier(sampled);
			double beta = computeAlpha(classifier, instances.numClasses(), labeled, unlabeled, confidence);
			if (beta <= 0) {
				break;
			}
			classifiers.add(classifier);
		} while (classifiers.size() < getNumIterations());
	}

	private double computeAlpha(WeightedClassifier classifier, int numClasses, Set<Instance> labeled, Set<Instance> unlabeled, Map<Instance, Confidence> mapConfidence) throws Exception {
		double num = 0;
		double dem = 0;
		for (Instance inst : labeled) {
			Confidence conf = mapConfidence.get(inst);
			double[] output = classifier.distributionForInstance(inst);
			int maxIndex = maxIndex(output);
			if (maxIndex == inst.classValue() && output[maxIndex] > 0) {
				num += conf.probability;
			} else {
				dem += conf.probability;
			}
		}
		for (Instance inst : unlabeled) {
			Confidence conf = mapConfidence.get(inst);
			double[] output = classifier.distributionForInstance(inst);
			int maxIndex = maxIndex(output);
			if (output[maxIndex] <= 0) {
				maxIndex = -1;
			}
			if (maxIndex == (int) inst.classValue()) {
				num += conf.p[maxIndex];
			} else {
				for (int i = 0; i < numClasses; i++) {
					if (i != maxIndex)
						dem += conf.p[i];
				}
			}
		}
		int k = numClasses;
		classifier.weight = Math.log(Math.pow(k - 1, 2) / k) * (Math.log(k - 1) + Math.log(num / dem));
		if (Double.isNaN(classifier.weight))
			classifier.weight = 1.0;
		return classifier.weight;
	}

	private Confidence confidence(Instance instance, Map<InstancePair, Double> similarities, Set<Instance> labeled, Set<Instance> unlabeled) throws Exception {
		int numClasses = instance.numClasses();
		double[] output = distributionForInstance(instance);
		double[] pl = new double[output.length];
		for (Instance inst : labeled) {
			int out = (int) inst.classValue();
			double sim = similarities.get(new InstancePair(inst, instance));
			pl[out] += sim * Math.exp(-output[out] / (double) (numClasses - 1.0));
		}
		double[] pu = new double[numClasses];
		double mul = Math.exp(1.0 / (double) (numClasses - 1.0));
		for (Instance inst : unlabeled) {
			double[] joutput = distributionForInstance(inst);
			double sim = similarities.get(new InstancePair(inst, instance));
			sim = sim * mul;
			for (int k = 0; k < numClasses; k++) {
				pu[k] += sim * Math.exp((joutput[k] - output[k]) / (double) (numClasses - 1.0));
			}
		}
		for (int i = 0; i < pl.length; i++) {
			pl[i] = pl[i] * constant2 + pu[i] * constant3;
		}
		Confidence conf = new Confidence(instance, pl);
		return conf;
	}

	private Confidence weighting(Instance instance, double[][] Y, int numClasses) throws Exception {
		double[] dist = distributionForInstance(instance);
		int y = (int) instance.classValue();
		double w = Math.exp(-dotProduct(dist, Y[y]) / (double) numClasses);
		w = w * constant1;
		Confidence conf = new Confidence(instance, dist, w);
		return conf;
	}

	private double[][] makeY(int numClasses) {
		double k = -1.0 / (double) (numClasses - 1);
		double[][] vectors = new double[numClasses][numClasses];
		for (int i = 0; i < numClasses; i++) {
			Arrays.fill(vectors[i], k);
			vectors[i][i] = 1.0;
		}
		return vectors;
	}

	private Map<Instance, Confidence> sampling(Instances sampled, Instances instances, Map<InstancePair, Double> similarities, Set<Instance> labeled, Set<Instance> unlabeled, Random rand) throws Exception {
		Map<Instance, Confidence> mapConfidence = new HashMap<>();
		int numClasses = instances.numClasses();
		double[][] Y = makeY(numClasses);
		double sumWeights = 0;
		for (Instance instance : labeled) {
			Confidence conf = weighting(instance, Y, numClasses);
			mapConfidence.put(instance, conf);
			sumWeights += conf.probability;
		}
		assert sumWeights > 0;
		for (Confidence conf : mapConfidence.values()) {
			conf.probability = conf.probability / sumWeights;
		}

		sumWeights = 0;
		for (Instance instance : unlabeled) {
			Confidence conf = confidence(instance, similarities, labeled, unlabeled);
			mapConfidence.put(instance, conf);
			instance.setClassValue(conf.pseudolabel);
			sumWeights += conf.probability;
		}
		for (Instance instance : unlabeled) {
			Confidence conf = mapConfidence.get(instance);
			conf.probability = conf.probability / sumWeights;
		}

		int numSampling = (int) Math.ceil(percentSampling * instances.numInstances());
		stochasticUniversalSampling(sampled, mapConfidence, unlabeled, numSampling, rand);
		return mapConfidence;
	}

	public void stochasticUniversalSampling(Instances output, Map<Instance, Confidence> mapConfidence, Set<Instance> unlabeled, int size, Random rand) {
		List<Instance> instances = new ArrayList<>(unlabeled.size());
		List<Double> culmulative = new ArrayList<>(unlabeled.size());
		double sum = 0;
		for (Instance instance : unlabeled) {
			Confidence conf = mapConfidence.get(instance);
			sum += conf.probability;
			culmulative.add(sum);
			instances.add(conf.instance);
		}
		if (sum <= 0) {
			return;
		}

		double dist = sum / (double) instances.size();
		double point = rand.nextDouble() * dist;

		int index = 0;
		for (int i = 0; i < size; i++) {
			while (culmulative.get(index) < point) {
				index += 1;
			}
			output.add(instances.get(index));
			point += dist;
		}
	}

	private void splitInstances(Instances instances, Set<Instance> labeled, Set<Instance> unlabeled) {
		assert instances.numInstances() > 0;
		this.labeled = new Instances(instances, 0);
		for (Instance instance : instances) {
			if (instance.classIsMissing()) {
				unlabeled.add(instance);
			} else {
				labeled.add(instance);
				this.labeled.add(instance);
			}
		}
		assert labeled.size() > 0;
	}

	private Map<InstancePair, Double> computeSimilarity(Instances instances) {
		int numInstances = instances.numInstances();
		Map<InstancePair, Double> similarities = new HashMap<>();
		for (int i = 0; i < numInstances; i++) {
			Instance instance = instances.get(i);
			for (int j = i + 1; j < numInstances; j++) {
				Instance other = instances.get(j);
				// cosine similarity, symmetric and [0,1]
				double sim = euclidianDistance(instance.toDoubleArray(), other.toDoubleArray());
				similarities.put(new InstancePair(instance, other), sim);
				similarities.put(new InstancePair(other, instance), sim);
			}
			similarities.put(new InstancePair(instance, instance), 0.0);
		}
		radialBasisFunction(similarities);
		return similarities;
	}

	public static double euclidianDistance(double[] a, double[] b) {
		double dist = 0;
		for (int i = 0; i < a.length - 1; i++) {
			dist += Math.pow(a[i] - b[i], 2);
		}
		return Math.sqrt(dist);
	}

	private void radialBasisFunction(Map<InstancePair, Double> similarities) {
		delta = percentile(deltaPercentile, similarities);
		assert delta > 0;
		for (Entry<InstancePair, Double> entry : similarities.entrySet()) {
			double sim = entry.getValue();
			entry.setValue(Math.exp(-Math.pow(sim, 2) / Math.pow(delta, 2)));
		}
	}

	private double percentile(double p, Map<InstancePair, Double> similarities) {
		List<Double> values = new ArrayList<>(similarities.size());
		for (Double d : similarities.values()) {
			values.add(d);
		}
		Collections.sort(values);
		return values.get((int) (p * values.size()));
	}

	private double dotProduct(double[] a, double[] b) {
		double r = 0;
		for (int i = 0; i < a.length; i++) {
			r += a[i] * b[i];
		}
		return r;
	}

	private int maxIndex(double[] values) {
		int max = 0;
		for (int i = 1; i < values.length; i++) {
			if (values[i] > values[max]) {
				max = i;
			}
		}
		return max;
	}

	/**
	 * Returns a string describing classifier.
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "A boosting framework for semi-supervised learning, termed as SemiBoost\n\n" + "For more information, see\n\n" + getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;
		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Jafar Tanha & Maarten van Someren & Hamideh Afsarmanesh");
		result.setValue(Field.YEAR, "2014");
		result.setValue(Field.TITLE, "Boosting for multiclass semi-supervised learning");
		result.setValue(Field.JOURNAL, "Pattern Recognition Letters");
		result.setValue(Field.VOLUME, "37");
		result.setValue(Field.PAGES, "63-77");
		return result;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return      the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disable(Capability.MISSING_VALUES);

		result.disableAllClasses();
		result.disable(Capability.BINARY_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.NOMINAL_CLASS);

		result.setMinimumNumberInstances(0);
		return result;
	}

	/**
	 * Returns an enumeration of the additional measure names 
	 * produced by the neighbour search algorithm, plus the chosen K in case
	 * cross-validation is enabled.
	 * 
	 * @return an enumeration of the measure names
	 */
	public Enumeration<String> enumerateMeasures() {
		Set<String> measures = new TreeSet<>();
		measures.add("Num of Classifiers");
		measures.add("Constant C1");
		measures.add("Constant C2");
		measures.add("Constant C3");
		measures.add("delta");
		for (int i = 0; i < classifiers.size(); i++) {
			measures.add("alpha:" + i + "");
		}
		return Collections.enumeration(measures);
	}

	/**
	 * Returns the value of the named measure from the 
	 * neighbour search algorithm, plus the chosen K in case
	 * cross-validation is enabled.
	 * 
	 * @param additionalMeasureName the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	public double getMeasure(String additionalMeasureName) {
		switch (additionalMeasureName) {
		case "Num of Classifiers":
			return classifiers.size();
		case "Constant C1":
			return constant1;
		case "Constant C2":
			return constant2;
		case "Constant C3":
			return constant3;
		case "delta":
			return delta;
		default:
			int index = Integer.parseInt(additionalMeasureName.substring(6, additionalMeasureName.length()));
			return classifiers.get(index).weight;
		}
	}

	/**
	 * Returns a description of this classifier.
	 *
	 * @return a description of this classifier as a string.
	 */
	public String toString() {
		if (classifiers == null) {
			return "SemiBoost: No model built yet.";
		}

		String output = "SemiBoost\n\n";
		output += "Base classifier: " + getClassifier().toString() + "\n\n";
		output += "Constant C1(labeled): " + constant1 + "\n";
		output += "Constant C2(labeled+unlabeled): " + constant2 + "\n";
		output += "Constant C3(unlabeled): " + constant3 + "\n";
		output += "Sampling Percent: " + percentSampling + "\n";
		output += "Delta Percentile: " + deltaPercentile + "\n";
		output += "Measures:\n\n";
		Enumeration<String> en = enumerateMeasures();
		while (en.hasMoreElements()) {
			String measure = en.nextElement();
			output += measure + ": " + getMeasure(measure) + "\n";
		}
		return output;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return		the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 10141 $");
	}

	private class WeightedClassifier implements Serializable {

		private static final long serialVersionUID = 1L;
		public Classifier classifier;
		public double weight;

		public WeightedClassifier() {
			try {
				classifier = AbstractClassifier.makeCopy(getClassifier());
			} catch (Exception e) {
				throw new Error(e);
			}
		}

		public void buildClassifier(Instances data) throws Exception {
			classifier.buildClassifier(data);
		}

		public double[] distributionForInstance(Instance instance) throws Exception {
			return classifier.distributionForInstance(instance);
		}
	}

	private class Confidence {
		public Instance instance;
		public double pseudolabel;
		public double[] p;
		public double probability;

		public Confidence(Instance instance, double[] p, double w) {
			this.instance = instance;
			this.p = p;
			this.probability = w;
			pseudolabel = Double.NaN;
		}

		public Confidence(Instance instance, double[] p) {
			this.p = p;
			this.instance = instance;
			int max = -1;
			int sec = -1;
			for (int i = 0; i < p.length; i++) {
				if (max == -1 || p[i] > p[max]) {
					sec = max;
					max = i;
				} else if (sec == -1 || p[i] > p[sec]) {
					sec = i;
				}
			}
			pseudolabel = max;
			this.probability = p[max] - p[sec];
		}

		public String toString() {
			return String.format("p=%s  prob=%.4f class=%d", Arrays.toString(p), probability, (int) pseudolabel);
		}
	}

	private class InstancePair {
		public Instance a;
		public Instance b;

		public InstancePair(Instance a, Instance b) {
			this.a = a;
			this.b = b;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + getEnclosingInstance().hashCode();
			result = prime * result + Objects.hash(a, b);
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj) {
				return true;
			}
			if (!(obj instanceof InstancePair)) {
				return false;
			}
			InstancePair other = (InstancePair) obj;
			if (!getEnclosingInstance().equals(other.getEnclosingInstance())) {
				return false;
			}
			return Objects.equals(a, other.a) && Objects.equals(b, other.b);
		}

		private MultiSemiAdaBoost getEnclosingInstance() {
			return MultiSemiAdaBoost.this;
		}
	}
	

	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(5);
		newVector.addElement(new Option("\tThe weight importance between the labeled data.\n\t(-1 for 1/#labeled).", "L", 1, "-L <importante>"));
		newVector.addElement(new Option("\tThe weight importance between the unlabeled data.\n\t(-1 for 1/#instances).", "U", 1, "-U <importante>"));
		newVector.addElement(new Option("\tThe weight importance between the labeled and the unlabeled data.\n\t(-1 for 1/(2 * #unlabeled)).", "I", 1, "-I <importante>"));
		newVector.addElement(new Option("\tThe percentage of unlabeled instances [0.0,1.0].\n\t(Default = 1.0)", "P", 1, "-P <percentage>"));
		newVector.addElement(new Option("\tThe percentile of the scale parameter controlling the spread of the radial basis function [0.0,1.0].\n\t(Default = 1.0)", "D", 1, "-D <percentile>"));
		newVector.addAll(Collections.list(super.listOptions()));
		return newVector.elements();
	}

	public void setOptions(String[] options) throws Exception {
		String constant1String = Utils.getOption('L', options);
		if (constant1String.length() > 0) {
			constant1 = Double.parseDouble(constant1String);
		} else {
			constant1 = 0.1;
		}
		String constant3String = Utils.getOption('U', options);
		if (constant3String.length() > 0) {
			constant3 = Double.parseDouble(constant3String);
		} else {
			constant3 = 0.1;
		}
		String constant2String = Utils.getOption('I', options);
		if (constant2String.length() > 0) {
			constant2 = Double.parseDouble(constant2String);
		} else {
			constant2 = 1;
		}
		String percentString = Utils.getOption('P', options);
		if (percentString.length() > 0) {
			percentSampling = Double.parseDouble(percentString);
		} else {
			percentSampling = 0.1;
		}
		String deltaString = Utils.getOption('D', options);
		if (deltaString.length() > 0) {
			deltaPercentile = Double.parseDouble(deltaString);
			if (deltaPercentile <= 0)
				deltaPercentile = 0.1;
		} else {
			deltaPercentile = 0.1;
		}
		super.setOptions(options);
		Utils.checkForRemainingOptions(options);
	}

	public String[] getOptions() {
		Vector<String> options = new Vector<String>();
		options.add("-L");
		options.add(Double.toString(constant1));
		options.add("-U");
		options.add(Double.toString(constant3));
		options.add("-I");
		options.add(Double.toString(constant2));
		options.add("-P");
		options.add(Double.toString(percentSampling));
		options.add("-D");
		options.add(Double.toString(deltaPercentile));
		Collections.addAll(options, super.getOptions());
		return options.toArray(new String[0]);
	}

	public String constantLabeledTipText() {
		return "The weight importance the labeled data.";
	}

	public void setConstantLabeled(double value) {
		constant1 = value;
	}

	public double getConstantLabeled() {
		return constant1;
	}
	
	public String constantUnlabeledTipText() {
		return "The weight importance the unlabeled data.";
	}

	public void setConstantUnlabeled(double value) {
		constant3 = value;
	}

	public double getConstantUnlabeled() {
		return constant3;
	}
	
	public String constantInteractionTipText() {
		return "The weight importance between the labeled and unlabeled data.";
	}

	public void setConstantInteraction(double value) {
		constant2 = value;
	}

	public double getConstantInteraction() {
		return constant2;
	}

	public String percentSamplingTipText() {
		return "The percentage of unlabeled instances (p>0).";
	}

	public void setPercentSampling(double value) {
		percentSampling = value;
	}

	public double getPercentSampling() {
		return percentSampling;
	}

	public String deltaPercentileTipText() {
		return "The percentile of the spread of the RBF (p>0).";
	}

	public void setDeltaPercentile(double value) {
		deltaPercentile = value;
	}

	public double getDeltaPercentile() {
		return deltaPercentile;
	}
}
