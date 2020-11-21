package moa.classifiers.eacd;

import java.util.Random;

import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.AbstractClassifier;

public class ReplicatorDynamics {

	protected Type[] types;

	private int matureTime;

	public ReplicatorDynamics(int numAttributes, int numInitialMembers, double rate, Random rand, int minSize, int maxSize, AbstractClassifier newMemberModel) {
		this.types = RandomSelection.select(numAttributes, numInitialMembers, rate, rand, minSize, maxSize, newMemberModel);
		this.matureTime = matureTime / 2;
	}

	public Type[] types() {
		return types;
	}

	public boolean isMature() {
		return matureTime <= 0;
	}

	public void train(Instances instances) {
		if (matureTime > 0) {
			matureTime -= 1;
			for (Type type : types) {
				type.grow();
			}
		} else {
			double sum = 0;
			for (Type type : types) {
				type.evaluate(instances);
				sum += type.cost();
			}
			sum = sum / (double) types.length;
			for (Type type : types) {
				if (type.cost() >= sum) {
					type.grow();
				} else {
					type.shrink();
				}
			}
		}
		for (Type type : types) {
			type.train(instances);
		}
	}
}
