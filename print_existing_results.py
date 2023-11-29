"""
This script is written after publication and is meant to provide a way
for people interested in this work, to quickly re-calculate our metrics
and see them in detail.
"""

import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

mapping_2304 = {
    "neutral": "neutral", # this is here for completeness
    "entailment": "entails",
    "contradiction": "contradicts"
}



def load_data(path):
    return pd.read_csv(
        path,
        encoding='UTF-8',
        sep=';'
    )



def print_metrics(print_title, results_dict):
    print('\n')
    print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')
    print(print_title)
    print('--- --- --- --- --- --- --- --- ---')
    print(f"{'accuracy:':<12} {results_dict['accuracy']}")
    print(f"{'precision:':<12} {results_dict['precision']}")
    print(f"{'recall:':<12} {results_dict['recall']}")
    print(f"{'fscore:':<12} {results_dict['fscore']}")
    print(f"{'support:':<12} {results_dict['support']}")
    print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def get_metrics(true, pred):
    accuracy = accuracy_score(true, pred)
    precision, recall, fscore, support = precision_recall_fscore_support(true, pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'support': support
    }



def chat_gpt_results():
    data = load_data(r'Output\chatGPT\(clean)AAE_1k_sample_with_predictions.csv').copy()

    predicted_relations = data['predicted_relation'].tolist()
    true_relations      = data['label'].tolist()

    results_dict = get_metrics(true_relations, predicted_relations)

    print_metrics(
        "Results for ChatGPT - Argument Annotated Essays - Random 1k sample",
        results_dict
    )
    print(
        f"True entailments:{true_relations.count('entails'):>6} || True contradictions:{true_relations.count('contradicts'):>6} || True neutral:{true_relations.count('neutral'):>6}"
    )
    print(
        f"Pred entailments:{predicted_relations.count('entails'):>6} || Pred contradictions:{predicted_relations.count('contradicts'):>6} || Pred neutral:{predicted_relations.count('neutral'):>6}")
    print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def routine_for_model(model):
    data = load_data(fr'Output\{model}\true_and_pred_relations(AAE_1k).csv').copy()

    predicted_relations = data['pred_label'].tolist()
    true_relations      = data['true_label'].tolist()

    results_dict = get_metrics(true_relations, predicted_relations)

    print_metrics(
        f"Results for {model} - Argument Annotated Essays - Random 1k sample",
        results_dict
    )
    print(
        f"True entailments:{true_relations.count('entails'):>6} || True contradictions:{true_relations.count('contradicts'):>6} || True neutral:{true_relations.count('neutral'):>6}"
    )
    print(
        f"Pred entailments:{predicted_relations.count('entails'):>6} || Pred contradictions:{predicted_relations.count('contradicts'):>6} || Pred neutral:{predicted_relations.count('neutral'):>6}")
    print('##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####')



def all_results():
    print("================================")
    print("===== RESULTS FOR CHAT GPT =====")
    print("================================")
    print()
    chat_gpt_results()
    print()
    print("=============================")
    print("===== RESULTS FOR BART =====")
    print("============================")
    print()
    routine_for_model( 'bart' )
    print()
    print("===============================")
    print("===== RESULTS FOR DEBERTA =====")
    print("===============================")
    print()
    routine_for_model( 'deberta' )
    print()
    print("===============================")
    print("===== RESULTS FOR MINILM2 =====")
    print("===============================")
    print()
    routine_for_model( 'miniLM2' )



# chat_gpt_results()
# routine_for_model(
#     'bart' # OPTIONS || 'bart', 'deberta', 'miniLM2'
# )

all_results()
