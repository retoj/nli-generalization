"""
This file and the mappings were used to clean the outputs and derive metrics initially.

They are included as part of the original work and they are not needed (if new predictions are not derived)


Usage here is complicated as every different model used or every different dataset requires
a different means of cleaning and application of all the correct mappings with care not to
distort the results. Our results for the paper can mostly be found in the "Output" folder
and it only required to use sklearn to re-derive our metrics.
"""

import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

##### ##### MAPPINGS FOR chat-GPT LABELS ##### #####

mapping_2304 = {
    "neutral": "neutral", # this is here for completeness
    "entailment": "entails",
    "contradiction": "contradicts"
}


### This is used for the chat-GPT output labels and is modeled to account for all 
### keys found in our specific queries!!!!
AAE_mapping = {
    "neutral.": "neutral",
    "neutral": "neutral",
    "Neutral.": "neutral",
    "Neutral": "neutral",
    "entailment": "entails",
    "Entailment": "entails",
    "entailment.": "entails",
    "Entailment.": "entails",
    "contradiction": "contradicts",
    "contradiction.": "contradicts",
    "Contradiction": "contradicts",
    "Contradiction.": "contradicts",
}

eval2304_mapping = {
    "neutral.": "neutral",
    "Neutral.": "neutral",
    "entailment.": "entailment",
    "entailment": "entailment",
    "The relation between the sentences is entailment.": "entailment",
    "Entailment.": "entailment",
    "Entailment": "entailment",
    "contradiction.": "contradiction",
    " contradiction.": "contradiction",
    "contradiction": "contradiction",
    "Contradiction": "contradiction",
    "The relation between the two sentences is contradiction.": "contradiction",
    "Contradiction.": "contradiction",
}

o_data = pd.read_csv(
    r"Output\bart\true_and_pred_relations(AAE_1k).csv",
    encoding='UTF-8', sep=';'
)

# remove this
o_data = o_data.dropna()

predicted_relations = o_data['pred_label'].tolist()
true_relations = o_data['true_label'].tolist()

#print(set(true_relations))

accuracy = accuracy_score(true_relations, predicted_relations)
precision, recall, fscore, support = precision_recall_fscore_support(true_relations, predicted_relations)

print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fscore: {fscore}")
print(f"support: {support}")
# make second label "contradiction"
print(true_relations.count("entailment"), true_relations.count("not_entailment"), true_relations.count("neutral"))
print(predicted_relations.count("entailment"), predicted_relations.count("not_entailment"), predicted_relations.count("neutral"))
print(set(true_relations))
