import spacy
import scispacy
from scispacy.candidate_generation import CandidateGenerator

class umls_map:
    def __init__(self,threshold):
        self.candidate_generator = CandidateGenerator(name="umls")
        self.kb = self.candidate_generator.kb
        self.threshold = threshold
    def __call__(self,disease):
        predicted = []
        batch_candidates = self.candidate_generator([disease], 30)
        for cand in batch_candidates[0]:
                score = max(cand.similarities)
                if score < self.threshold or self.kb.cui_to_entity[cand.concept_id].definition==None:
                    continue
                else:
                    predicted.append((cand.concept_id, score))
        if len(predicted)>0:
                sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[1])[0]
                #('C0007131', 0.9999999403953552)
                name = self.kb.cui_to_entity[sorted_predicted[0]].canonical_name
                return name
        else:
              return disease