package moa.classifiers.eacd;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import moa.classifiers.AbstractClassifier;

public class RandomSelection {

	public static Type[] select(int numAttributes, int mSteps, Random rand, int minSize, int maxSize, AbstractClassifier newMemberModel) {
		Type[] types = new Type[mSteps];
		for (int i = 0; i < types.length; i++) {
			types[i] = selectAttributes(numAttributes, rand.nextDouble(), rand, minSize, maxSize, newMemberModel);
		}
		return types;
	}
	
	public static Type[] select(int numAttributes, int mSteps, double pRate, Random rand, int minSize, int maxSize, AbstractClassifier newMemberModel) {
		Type[] types = new Type[mSteps];
		for (int i = 0; i < types.length; i++) {
			types[i] = selectAttributes(numAttributes, pRate, rand, minSize, maxSize, newMemberModel);
		}
		return types;
	}

	protected static Type selectAttributes(int numAttributes, double rate, Random rand, int minSize, int maxSize, AbstractClassifier newMemberModel) {
		int size = (int) (numAttributes * rate);
		int vetSize = numAttributes;
		List<Integer> atts = new ArrayList<>();
		for (int i = 0; i < numAttributes; i++) {
			atts.add(i);
		}
		boolean[] attributes = new boolean[numAttributes];
		for (int i = 0; i < size; i++) {
			int att = atts.remove(rand.nextInt(vetSize--));
			attributes[att] = true;
		}
		Type type = new Type(attributes, minSize, maxSize, newMemberModel);
		return type;
	}
}
