package moa.classifiers.eacd;

import java.util.ArrayList;
import java.util.List;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.AbstractClassifier;
import moa.core.Utils;

public class Type {

	public final int minMembers;

	public final int maxMembers;

	public final boolean[] attributes;

	protected List<AbstractClassifier> members;

	protected List<Double> accuracy;

	private AbstractClassifier newMemberModel;

	public Type(boolean[] attributes, int minMembers, int maxMembers, AbstractClassifier newMemberModel) {
		this.newMemberModel = newMemberModel;
		this.minMembers = minMembers;
		this.maxMembers = maxMembers;
		this.attributes = attributes;
		this.accuracy = new ArrayList<>();
		this.members = new ArrayList<>();
		grow();
	}
	
	public String toString () {
		return accuracy.toString();
	}

	public int size() {
		return members.size();
	}

	public void train(Instances block) {
		for (int i = 0; i < members.size(); i++) {
			AbstractClassifier c = members.get(i);
			for (int j = 0; j < block.numInstances(); j++) {
				Instance inst = block.get(j);
				inst = filter(inst, attributes);
				c.trainOnInstance(inst);
			}
		}
	}

	public double classify(Instances block) {
		double acc = 0;
		for (int i = 0; i < members.size(); i++) {
			int count = 0;
			for (int j = 0; j < block.numInstances(); j++) {
				Instance inst = block.get(j);
				inst = filter(inst, attributes);
				AbstractClassifier c = members.get(i);
				c.getPredictionForInstance(inst);
				int y = Utils.maxIndex(c.getVotesForInstance(inst));
				int real = (int) inst.classValue();
				if (y == real) {
					count += 1;
				}
			}
			acc += count / (double) block.numInstances();
		}
		return acc / (double) block.numInstances();
	}

	public void evaluate(Instances block) {
		for (int i = 0; i < members.size(); i++) {
			int count = 0;
			for (int j = 0; j < block.numInstances(); j++) {
				Instance inst = block.get(j);
				inst = filter(inst, attributes);
				AbstractClassifier c = members.get(i);
				c.getPredictionForInstance(inst);
				int y = Utils.maxIndex(c.getVotesForInstance(inst));
				int real = (int) inst.classValue();
				if (y == real) {
					count += 1;
				}
			}
			accuracy.set(i, count / (double) block.numInstances());
		}
	}

	public double[][] distribution(Instance inst) {
		inst = filter(inst, attributes);
		double[][] vet = new double[members.size()][];
		for (int i = 0; i < members.size(); i++) {
			vet[i] = members.get(i).getVotesForInstance(inst);
		}
		return vet;
	}

	public void shrink() {
		if (members.size() > 1) {
			double min = 1.0;
			int minIndex = -1;
			for (int i = 0; i < accuracy.size(); i++) {
				double acc = accuracy.get(i);
				if (acc < min || minIndex == -1) {
					minIndex = i;
					min = acc;
				}
			}
			members.remove(minIndex);
			accuracy.remove(minIndex);
		}
	}

	public void grow() {
		if (members.size() >= maxMembers) {
			shrink();
		}
		members.add(createNewTree());
		accuracy.add(1.0);
	}

	public double cost() {
		double avg = 0;
		for (double acc : accuracy) {
			avg += acc;
		}
		return avg / (double) accuracy.size();
	}

	private AbstractClassifier createNewTree() {
		AbstractClassifier classifier = (AbstractClassifier) newMemberModel.copy();
		classifier.resetLearning();
		return classifier;
	}

	private Instance filter(Instance _instance, boolean[] active) {
		int numActives = 0;
		for (int i = 0; i < active.length; i++) {
			if (active[i]) {
				numActives += 1;
			}
		}
		if (numActives == active.length) {
			return _instance;
		}

		Instances instances = new Instances(_instance.dataset(), 0);
		ArrayList<Attribute> attributes = new ArrayList<>();
		int numAttributes = _instance.numAttributes();
		double[] values = new double[numActives + 1];
		for (int i = 0, count = 0; i < numAttributes; i++) {
			if (i == numAttributes - 1 || active[i]) {
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
