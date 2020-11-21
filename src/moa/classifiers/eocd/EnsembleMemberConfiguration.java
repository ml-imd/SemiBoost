package moa.classifiers.eocd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class EnsembleMemberConfiguration {

	protected List<Member> members;
	protected MemberClassifier[] classifiers;

	private class Member implements Comparable<Member> {
		int type;
		int hyper;
		Set<Integer> removedAtt;
		double weight;

		public Member(int type, int hyper, Set<Integer> removedAtt, double weight) {
			this.type = type;
			this.hyper = hyper;
			this.removedAtt = new HashSet<>(removedAtt);
			this.weight = weight;
		}

		@Override
		public int compareTo(Member other) {
			int cmp = 0;
			if (type < other.type) {
				cmp = -1;
			} else if (type > other.type) {
				cmp = 1;
			}
			if(cmp == 0) {
				if (hyper < other.hyper) {
					cmp = -1;
				} else if (hyper > other.hyper) {
					cmp = 1;
				}	
			}
			return cmp;
		}
	}

	public EnsembleMemberConfiguration(MemberClassifier[] classifiers) {
		this.members = new ArrayList<>();
		this.classifiers = classifiers;
		for (MemberClassifier member : classifiers) {
			if (member.isActive) {
				members.add(new Member(member.type, member.hyperSet, member.removedAttributes(), member.weight));
			}
		}
	}

	public void configure(Solve solve) {
		Arrays.fill(solve.encode, 0.0);
		boolean[] used = new boolean[classifiers.length];
		Problem problem = (Problem) solve.problem;
		for (Member member : members) {
			int maxIndex = -1;
			int maxSim = -1;

			for (int i = 0; i < classifiers.length; i++) {
				if (!used[i] && member.type == classifiers[i].type) {
					int sim = similarity(classifiers[i], member);
					if (sim > maxSim) {
						maxSim = sim;
						maxIndex = i;
					}
				}
			}

			if (maxIndex != -1) {
				used[maxIndex] = true;
				problem.setWeight(solve, maxIndex, member.weight);
			}
		}

	}

	private int similarity(MemberClassifier classifier, Member member) {
		int sim = 0;
		if (classifier.type == member.type) {
			for (Integer att : classifier.removedAttributes()) {
				if (member.removedAtt.contains(att)) {
					sim += 1;
				}
			}
		}
		return sim;
	}
}