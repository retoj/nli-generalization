from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli')
tokenizer = AutoTokenizer.from_pretrained("ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli")

### The label mapping has been produced upon close inspection
### of the model outputs translated to our requirements
label_mapping = ["contradicts", "entails", "neutral"]



def bart_predict_relations(premise: str, hypothesis: str):

    features = tokenizer([premise], [hypothesis], padding=True, return_tensors="pt")
    model.eval()

    with torch.no_grad():
        scores = model(**features).logits
    # print(f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label_mapping[scores.argmax()]}\n##############################################################################   ")
    return label_mapping[scores.argmax()]
