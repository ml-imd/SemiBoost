package moa.classifiers.eocd;

import java.io.Serializable;

public class StopCondition implements Serializable {

	private static final long serialVersionUID = 1L;

	private class Events implements SingleConditionCallback, Serializable {

		private static final long serialVersionUID = 1L;
		public int numMinBound = 0;
		public int numMaxBound = 0;

		@Override
		public void minimumBound() {
			numMinBound += 1;
			isRunning = isRunning && (numMinBound < 4 || numMaxBound == 0);
			assert (numMinBound <= 4) : "Num Min Bound " + numMinBound;
		}

		@Override
		public void maximumBound() {
			numMaxBound += 1;
			isRunning = isRunning && numMinBound < 4;
			assert (numMaxBound <= 4) : "Num Max Bound " + numMaxBound;
		}

		public void reset() {
			numMinBound = 0;
			numMaxBound = 0;
		}
	}

	private SingleCondition evals;
	private SingleCondition steps;
	private SingleCondition updts;
	private SingleCondition times; // in milliseconds

	private boolean isRunning;
	private long startTime;
	private final Events call = new Events();

	public StopCondition(int minEvals, int maxEvals, int minSteps, int maxSteps, int maxNoUpdt, int minSecs, int maxSecs) {
		assert minEvals >= 0;
		assert minSteps >= 0;
		assert minSecs >= 0;
		evals = new SingleCondition(minEvals, maxEvals == -1 ? Integer.MAX_VALUE : maxEvals, call);
		steps = new SingleCondition(minSteps, maxSteps == -1 ? Integer.MAX_VALUE : maxSteps, call);
		updts = new SingleCondition(0, maxNoUpdt == -1 ? Integer.MAX_VALUE : maxNoUpdt, call);
		times = new SingleCondition(minSecs * 1000, maxSecs == -1 ? Integer.MAX_VALUE : maxSecs * 1000, call);
	}

	public StopCondition(int maxEvals, int maxSteps, int maxNoUpdts, int maxTime) {
		this(0, maxEvals, 0, maxSteps, maxNoUpdts, 0, maxTime);
	}

	public StopCondition(int minSteps, int maxSteps, int maxNoUpdt) {
		this(0, -1, minSteps, maxSteps, maxNoUpdt, 0, -1);
	}

	public StopCondition(int minEvals, int maxEvals) {
		this(minEvals, maxEvals, 0, -1, -1, 0, -1);
	}

	public StopCondition(int maxSeconds) {
		this(0, -1, 0, -1, -1, 0, maxSeconds);
	}

	public void reset() {
		call.reset();
		isRunning = false;
		evals.reset();
		updts.reset();
		steps.reset();
		times.reset();
	}

	private void updateTime() {
		int diffTime = (int) (System.currentTimeMillis() - startTime);
		times.count(0);
		times.add(diffTime);
	}

	public boolean isRunning() {
		return isRunning;
	}

	public void start() {
		reset();
		startTime = System.currentTimeMillis();
		isRunning = true;

		new Thread() {
			@Override
			public void run() {
				while (!times.maxBound() && isRunning()) {
					updateTime();
					try {
						Thread.sleep(250);
					} catch (InterruptedException e) {
						e.printStackTrace();
						break;
					}
				}
			}
		}.start();
	}

	public void stop() {
		if (isRunning) {
			updateTime();
			isRunning = false;
		}
	}

	public void update() {
		if (isRunning) {
			if (updts.maxBound()) {
				call.numMaxBound -= 1;
			}
			if (updts.minBound()) {
				call.numMinBound -= 1;
			}
			updts.reset();
		}
	}

	public void iteration() {
		if (isRunning) {
			steps.add(1);
			updts.add(1);
		}
	}

	public void evaluation() {
		if (isRunning) {
			evals.add(1);
		}
	}

	public int performedIterationsWhitoutUpdate() {
		return updts.count();
	}

	public int performedIterations() {
		return steps.count();
	}

	public int performedEvaluations() {
		return evals.count();
	}

	public int performedSeconds() {
		return times.count() / 1000;
	}

	public int maxAllowedSeconds() {
		return times.maxValue == Integer.MAX_VALUE ? -1 : times.maxValue / 1000;
	}

	public int maxAllowedIterations() {
		return steps.maxValue == Integer.MAX_VALUE ? -1 : steps.maxValue;
	}

	public int maxAllowedEvaluations() {
		return evals.maxValue == Integer.MAX_VALUE ? -1 : evals.maxValue;
	}

	public int maxAllowedIterationsWhitoutUpdate() {
		return updts.maxValue == Integer.MAX_VALUE ? -1 : updts.maxValue;
	}

	@Override
	public String toString() {
		StringBuilder str = new StringBuilder(1000);
		str.append("StopCondition:");

		String[] labels = { "Steps", "NoUpdate", "Evaluations", "Seconds" };
		int[] max = { maxAllowedIterations(), maxAllowedIterationsWhitoutUpdate(), maxAllowedEvaluations(), maxAllowedSeconds() };
		int[] cur = { steps.count(), updts.count(), evals.count(), times.count() / 1000 };

		for (int i = 0; i < labels.length; i++) {
			if (max[i] > 0) {
				str.append(String.format(" %s(%d/%d)", labels[i], cur[i], max[i]));
			} else {
				str.append(String.format(" %s(%d)", labels[i], cur[i]));
			}
		}

		return str.toString();
	}
}


interface SingleConditionCallback {

	default void minimumBound() {
	}

	default void maximumBound() {
	}
}

class SingleCondition implements Serializable {

	private static final long serialVersionUID = 1L;
	
	private int count;
	private boolean min;
	private boolean max;

	public final int minValue;
	public final int maxValue;
	public final SingleConditionCallback call;

	public SingleCondition(int minValue, int maxValue, SingleConditionCallback call) {
		assert minValue <= maxValue;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.call = call;
		reset();
	}

	public void reset() {
		min = false;
		max = false;
		count = 0;
		add(0);
	}

	public int count() {
		return count;
	}

	public void count(int count) {
		this.count = count;
	}

	public void add(int value) {
		count += value;
		if (!min && count >= minValue) {
			min = true;
			call.minimumBound();
		}
		if (!max && count >= maxValue) {
			max = true;
			call.maximumBound();
		}
	}

	public boolean maxBound() {
		return max;
	}

	public boolean minBound() {
		return min;
	}
}

