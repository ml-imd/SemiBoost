package moa.classifiers.eocd;

import java.util.Arrays;
import java.util.Set;

public class Solve implements Comparable<Solve> {

	public static final double MIN_VALUE = 0.5;
	public static final double MIN_VALUE_ACTIVE = 0.5;
	
	public final double[] encode;
	public final Problem problem;
	
	private final double[] objective;
	private boolean evaluated;

	public Solve(Problem problem) {
		this.problem = problem;
		objective = new double[problem.numObjectives];
		encode = new double[problem.sizeEncode];
		evaluated(false);
	}
	
	public void repair() {
		Problem problem = (Problem) this.problem;
		Set<Integer> active = problem.activeClassifiers();
		for (int i = 0; i < encode.length; i += 2) {
			if (active.contains(i)) {
				if (encode[i] < MIN_VALUE) {
					encode[i] = 0.0;
				}
			} else {
				encode[i] = 0.0;
			}
		}
	}

	public boolean isActive(int member) {
		return encode[member] > MIN_VALUE;
	}

	public double weight(int member) {
		return isActive(member) ? encode[member] : 0.0;
	}

	public void weight(int member, double weight) {
		encode[member] = weight;
	}

	@Override
	public String toString() {
		return ((Problem) this.problem).createRepresentation(this);
	}

	public Solve clone() {
		Solve solve = new Solve(problem);
		solve.copy(this);
		return solve;
	}

	public boolean isValid() {
		return evaluated;
	}

	public void evaluated(boolean eval) {
		evaluated = eval;
		if (!eval) {
			Arrays.fill(objective, 1.0);
		}
	}

	public boolean evaluated() {
		return evaluated;
	}

	public double cost() {
		return objective[0];
	}

	public void cost(double cost) {
		this.objective[0] = cost;
	}

	public double objective(int index) {
		assert index < objective.length;
		return objective[index];
	}

	public void objective(int index, double value) {
		assert index < objective.length;
		objective[index] = value;
	}

	public double[] objectives() {
		return objective;
	}

	public void objectives(double[] objectives) {
		assert objectives.length == objective.length;
		System.arraycopy(objectives, 0, this.objective, 0, this.objective.length);
	}

	public int numObjectives() {
		return objective.length;
	}

	public boolean noDominated(Solve solve) {
		return !(dominates(solve) || solve.dominates(this) || equals(solve));
	}

	public boolean dominates(Solve solve) {
		boolean allNotWorse = true;
		boolean oneStrictlyBetter = false;
		for (int i = 0; i < objective.length; i++) {
			allNotWorse = allNotWorse && objective(i) <= solve.objective(i);
			oneStrictlyBetter = oneStrictlyBetter || objective(i) < solve.objective(i);
		}
		return allNotWorse && oneStrictlyBetter;
	}

	@Override
	public int hashCode() {
		int hash = 1237;
		if (evaluated) {
			hash = Arrays.hashCode(objective);
			hash = 31 * hash + 1231;
			return hash;
		}
		return hash;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (!(o instanceof Solve)) {
			return false;
		}
	
		Solve solve = (Solve) o;
		if (evaluated && solve.evaluated) {
			return Arrays.equals(objective, solve.objective);
		}
		return !(evaluated || solve.evaluated);
	}
	
	@Override
	public int compareTo(Solve solve) {
		int cmp = 0;
		if (evaluated() && solve.evaluated()) {
			cmp = 0;
			for (int i = 0; i < numObjectives() && cmp == 0; i++) {
				cmp = Double.compare(objective(i), solve.objective(i));
			}
		} else if (evaluated() || solve.evaluated()) {
			cmp = evaluated() ? -1 : +1;
		}
		return cmp;
	}

	public void copy(Solve solve) {
		System.arraycopy(solve.encode, 0, encode, 0, encode.length);
		System.arraycopy(solve.objectives(), 0, objectives(), 0, solve.numObjectives());
		evaluated(solve.evaluated());
	}

	public boolean equalsEncode(Solve solve) {
		boolean eq = true;
		for (int i = 0; i < encode.length && eq; i++) {
			eq = Double.compare(encode[i], solve.encode[i]) == 0;
		}
		return eq;
	}
}
