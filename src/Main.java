import moa.classifiers.HeterogeneousDynamicWeightedMajority;
import moa.evaluation.preview.LearningCurve;
import moa.streams.InstanceStream;
import moa.streams.generators.SEAGenerator;
import moa.tasks.EvaluatePrequential;

public class Main {

	public static void main(String[] args) {
		InstanceStream stream = new SEAGenerator();
		//EnsembleOptimizationForConceptDrift classifier = new EnsembleOptimizationForConceptDrift();
		//EvolutionaryAdaptationToConceptDrifts classifier = new EvolutionaryAdaptationToConceptDrifts();
		HeterogeneousDynamicWeightedMajority classifier = new HeterogeneousDynamicWeightedMajority();
		EvaluatePrequential eval = new EvaluatePrequential();
		eval.sampleFrequencyOption.setValue(100);
		eval.instanceLimitOption.setValue(10000);
		eval.streamOption.setCurrentObject(stream);
		eval.learnerOption.setCurrentObject(classifier);
		eval.prepareForUse();
		LearningCurve curve = (LearningCurve) eval.doTask();

		System.out.println("classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),Kappa M Statistic (percent),evaluation time (cpu seconds),model cost (RAM-Hours)");
		int[] index = { 4, 5, 6, 7, 1, 2 };
		for (int i = 0; i < curve.numEntries(); i++) {
			StringBuilder str = new StringBuilder(1000);
			for (int j = 0; j < index.length; j++) {
				str.append(String.format("%f,", curve.getMeasurement(i, index[j])));
			}
			System.out.println(str.toString());
		}
		System.out.close();
	}

}
