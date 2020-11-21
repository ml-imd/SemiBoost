package moa.classifiers.eocd;

import java.util.ArrayList;
import java.util.Random;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.AbstractClassifier;
import moa.core.Utils;

public class MemberClassifier {

	final AbstractClassifier classifier;

	public final int type;

	public final int hyperSet;

	public double weight;

	public boolean isActive;

	public boolean isHidden;

	private final SortedSet<Integer> removedAtt;

	private Random classifierRandom;

	private int numAttributes;
	
	private double trained;
	
	private double miss;

	public MemberClassifier(AbstractClassifier classifier, int hyperSet, Random random, int numAttributes) {
		this.classifier = classifier;
		classifier.resetLearning();
		this.weight = 1.0;
		this.isActive = false;
		this.isHidden = false;
		this.hyperSet = hyperSet;
		this.type = classifier.getClass().getSimpleName().hashCode();
		this.classifierRandom = random;
		this.numAttributes = numAttributes;
		this.removedAtt = new TreeSet<Integer>();
		startFilter(classifierRandom.nextDouble());
	}

	public SortedSet<Integer> removedAttributes() {
		return new TreeSet<>(removedAtt);
	}

	private void startFilter(double percent) {

		// use poisson distribution to determine the number of attributes
		for (int i = 0; i < numAttributes - 1; i++) { // -1 for class
			if (classifierRandom.nextDouble() < percent) {
				removedAtt.add(i);
			}
		}
		if (removedAtt.size() == numAttributes - 1) { // at least 1 attribute
			removedAtt.remove(classifierRandom.nextInt(numAttributes - 1));
		}
	}

	public boolean correctlyClassifies(Instance instance) {
		Instance inst = filter(instance, removedAtt);
		return classifier.correctlyClassifies(inst);
	}
	
	public double accuracy () {
		return trained == 0 ? 1.0 : (trained - miss) / (double) trained;
	}

	public void train(Instance instance) {
		trained += 1;
		int y = Utils.maxIndex(distribution(instance));
		if(y != (int)instance.classValue()) {
			miss += 1;
		}
		Instance inst = filter(instance, removedAtt);
		classifier.trainOnInstance(inst);
	}

	public double[] distribution(Instance instance) {
		Instance inst = filter(instance, removedAtt);
		return classifier.getVotesForInstance(inst);
	}

	@Override
	public String toString() {
		return String.format("(%c,%d,%s,%f)", classifier.getClass().getSimpleName().charAt(0), hyperSet, removedAtt.toString(), weight);
	}

	private Instance filter(Instance _instance, Set<Integer> remove) {
		if (remove.isEmpty()) {
			return _instance;
		}

		Instances instances = new Instances(_instance.dataset(), 0);
		ArrayList<Attribute> attributes = new ArrayList<>();
		int numAttributes = _instance.numAttributes();
		double[] values = new double[numAttributes - remove.size()];
		for (int i = 0, count = 0; i < numAttributes; i++) {
			if (!remove.contains(i)) {
				values[count++] = _instance.value(i);
				// Attribute att = cloneAttribute(instances.attribute(i));
				Attribute att = instances.attribute(i);
				attributes.add(att);
			}
		}

		Instances removedAttributes = new Instances("", attributes, 1);
		removedAttributes.setClassIndex(removedAttributes.numAttributes() - 1);

		Instance instance = new DenseInstance(1.0, values);
		instance.setDataset(removedAttributes);
		return instance;
	}
}
