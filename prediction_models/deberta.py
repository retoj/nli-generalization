from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/nli-deberta-v3-large')
### The label mapping has been produced upon close inspection
### of the model outputs translated to our requirements
label_mapping = ['contradiction', 'entailment', 'neutral']



def deberta_predict_relations(premise: str, hypothesis: str):

    scores = model.predict([(premise, hypothesis)])
    # print(f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label_mapping[scores.argmax()]}\n##############################################################################   ")
    return label_mapping[scores.argmax()]
