"""
This script is used to evaluate the datasets on the different model
architectures defined in the imports bellow.
"""

import pickle
import pandas as pd
from prediction_models.bart import bart_predict_relations
from prediction_models.deberta import deberta_predict_relations
from prediction_models.minilm2 import miniLM2_predict_relations



def return_data():
    return pd.read_csv(
        r"Data\AAE_1k_sample.csv",
        encoding='UTF-8',
        sep=';'
    )
        


all_data_with_predictions = []

RTE_mapping = {
    "entailment": "entailment",
    "contradiction": "not_entailment",
    "neutral": "not_entailment"
}

data_copy = return_data().copy()
# print(data_copy)
### Change the function call in pred_label to use another model
### OPTIONS:
### bart_predict_relations || deberta_predict_relations || predict_ft_deberta || predict_ft_minilm2 || miniLM2_predict_relations
for index, row in data_copy.iterrows():
    pred_label = deberta_predict_relations(row["sentence1"], row["sentence2"])
    all_data_with_predictions.append([row["sentence1"], row["sentence2"], row["label"], pred_label])


##### ##### UNCOMMENT TO STORE PREDICTIONS ##### #####


### Creating dataframe - clean version of data
# new_df = pd.DataFrame(all_data_with_predictions, columns=["sentence1", "sentence2", "true_label", "pred_label"])

# new_df.to_csv(
#     r"Output\deberta\true_and_pred_relations(AAE_1k).csv",
#     encoding='UTF-8',
#     sep=';'
# )

### Raw version - dump into a pickle
# with open(r"Output\deberta\true_and_pred_relations(AAE_1k).pkl", 'wb') as f_out:
#     pickle.dump(all_data_with_predictions, f_out)
