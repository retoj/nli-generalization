from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
label_mapping = ['contradiction', 'entailment', 'neutral']


def miniLM2_predict_relations(premise: str, hypothesis: str):

    scores = model.predict([premise, hypothesis])
    # print(f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label_mapping[scores.argmax()]}\n##############################################################################   ")
    return label_mapping[scores.argmax()]
