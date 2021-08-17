package weka.classifiers.semi;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
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

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.converters.ArffLoader;

public class SemiBoost extends RandomizableIteratedSingleClassifierEnhancer implements OptionHandler, TechnicalInformationHandler, AdditionalMeasureProducer {

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain command line options (see setOptions)
	 * @throws Exception 
	 */
	public static void main(String[] argv) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File("Vehicle23.arff"));
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		double acc = 0;
		int n = 100;
		for (int i = 0; i < n; i++) {

			Random rand = new Random(i);
			instances.randomize(rand);

			Instances test = new Instances(instances, 0, instances.numInstances() / 2);
			Instances train = new Instances(instances, instances.numInstances() / 2, instances.numInstances() - instances.numInstances() / 2);

			int count0 = 0;
			int count1 = 0;
			int limit = (int) (train.numInstances() * 0.05);
			for (Instance instance : train) {
				if (instance.classValue() == 0) {
					if (count0 < limit) {
						count0 += 1;
					} else {
						instance.setClassMissing();
					}
					if (count1 < limit) {
						count1 += 1;
					} else {
						instance.setClassMissing();
					}
				}
			}

			SemiBoost classifier = new SemiBoost();
			classifier.setSeed(i);
			classifier.buildClassifier(new Instances(train));
			Evaluation eval = new Evaluation(new Instances(train));
			eval.evaluateModel(classifier, test);
			acc += eval.pctCorrect();
			System.out.println(eval.pctCorrect());
		}
		System.out.println("AVG " + acc / (double) n);

	}

	private static final long serialVersionUID = 1;

	private List<WeightedClassifier> classifiers;
	protected double constant; // set default #l/#u as -1
	protected double percentSampling = 0.1;
	protected double deltaPercentile = 0.1; // 10 a 20
	private double delta;
	private Instances labeled;

	public SemiBoost() {
		setClassifier(new J48());
		setNumIterations(10);
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if an error occurred during the prediction
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] output = new double[2]; // only binary
		double h = combinedClassification(instance);
		if (h < 0) {
			output[0] = 1.0;
		} else {
			output[1] = 1.0;
		}
		return output;
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double h = combinedClassification(instance);
		return h <= 0 ? 0 : 1;
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
		constant = unlabeled.size() > 0 ? labeled.size() / (double) unlabeled.size() : 1.0;

		classifiers = new ArrayList<>(getNumIterations());
		Map<InstancePair, Double> similarities = computeSimilarity(instances);

		Random rand = new Random(getSeed());
		do {
			Instances sampled = new Instances(this.labeled);
			Map<Instance, Confidence> confidence = sampling(sampled, instances, similarities, labeled, unlabeled, rand);
			WeightedClassifier classifier = new WeightedClassifier();
			classifier.buildClassifier(sampled);
			if (computeAlpha(classifier, unlabeled, confidence) <= 0) {
				break;
			}
			classifiers.add(classifier);
		} while (classifiers.size() < getNumIterations());
	}

	private double computeAlpha(WeightedClassifier classifier, Set<Instance> unlabeled, Map<Instance, Confidence> mapConfidence) throws Exception {
		double num = 0;
		double dem = 0;
		for (Instance instance : unlabeled) {
			Confidence conf = mapConfidence.get(instance);
			double output = classifier.classifyInstance(instance) == 0.0 ? -1 : 1;
			if (output == 1) {
				num += conf.p;
				dem += conf.q;
			} else {
				num += conf.q;
				dem += conf.p;
			}
		}
		classifier.weight = 0.25 * Math.log(num / dem);
		if (Double.isNaN(classifier.weight))
			classifier.weight = 1.0;
		return classifier.weight;
	}

	private Confidence confidence(Instance instance, Map<InstancePair, Double> similarities, Set<Instance> labeled, Set<Instance> unlabeled) throws Exception {
		double p = 0;
		double q = 0;
		double output = combinedClassification(instance);
		for (Instance inst : unlabeled) {
			double sim = similarities.get(new InstancePair(instance, inst));
			double joutput = combinedClassification(inst);
			p += sim * Math.exp(joutput - output);
			q += sim * Math.exp(output - joutput);
		}
		p = (constant * p) / 2.0;
		q = (constant * q) / 2.0;

		for (Instance inst : labeled) {
			double sim = similarities.get(new InstancePair(instance, inst));
			if (inst.classValue() == 1) {
				p += sim * Math.exp(-2 * output);
			} else {
				q += sim * Math.exp(2 * output);
			}
		}
		return new Confidence(instance, p, q);
	}

	private Map<Instance, Confidence> sampling(Instances sampled, Instances instances, Map<InstancePair, Double> similarities, Set<Instance> labeled, Set<Instance> unlabeled, Random rand) throws Exception {
		Map<Instance, Confidence> mapConfidence = new HashMap<>();
		for (Instance instance : unlabeled) {
			Confidence conf = confidence(instance, similarities, labeled, unlabeled);
			mapConfidence.put(instance, conf);
			instance.setClassValue(conf.sign);
		}
		int numSampling = unlabeled.size() > 0 ? (int) Math.ceil(percentSampling * unlabeled.size()) : instances.numInstances();
		stochasticUniversalSampling(sampled, mapConfidence.values(), numSampling, rand);
		return mapConfidence;
	}

	public void stochasticUniversalSampling(Instances output, Collection<Confidence> confidences, int size, Random rand) {
		List<Instance> instances = new ArrayList<>(confidences.size());
		List<Double> culmulative = new ArrayList<>(confidences.size());
		double sum = 0;
		for (Confidence conf : confidences) {
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

	private double combinedClassification(Instance instance) throws Exception {
		double h = 0;
		for (WeightedClassifier classifier : classifiers) {
			int output = classifier.classifyInstance(instance) == 0.0 ? -1 : 1;
			h += classifier.weight * output;
		}
		return h;
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
		result.setValue(Field.AUTHOR, "P. K. Mallapragada, R. Jin, A. K. Jain and Y. Liu");
		result.setValue(Field.YEAR, "2009");
		result.setValue(Field.TITLE, "SemiBoost: Boosting for Semi-Supervised Learning");
		result.setValue(Field.JOURNAL, "IEEE Transactions on Pattern Analysis and Machine Intelligence");
		result.setValue(Field.VOLUME, "31");
		result.setValue(Field.PAGES, "2000-2014");
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
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.BINARY_CLASS);

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
		measures.add("Constant C");
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
		case "Constant C":
			return constant;
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
		output += "Constant: " + constant + "\n";
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

		public double classifyInstance(Instance instance) throws Exception {
			return classifier.classifyInstance(instance);
		}
	}

	private class Confidence {
		public Instance instance;
		public double p;
		public double q;
		public double probability;
		public double sign;

		public Confidence(Instance instance, double p, double q) {
			this.instance = instance;
			this.p = p;
			this.q = q;
			this.probability = Math.abs(p - q);
			this.sign = p > q ? 1 : 0;
		}

		public String toString() {
			return String.format("p=%.4f q=%.4f prob=%.4f sign=%+d", p, q, probability, sign > 0 ? 1 : -1);
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

		private SemiBoost getEnclosingInstance() {
			return SemiBoost.this;
		}
	}
}
