import pickle
import pandas as pd
import spacy
import random


### Function to clean stopwords and newline tokens
def clean_sent(sent):
    sent = sent.replace("\n", "")
    sent = sent.replace('"', '')
    sent = sent.replace("'", "")
    return sent

### Load pretrained spacy model
nlp = spacy.load("en_core_web_sm")

### Load the original annotations
with open(r"Data\annotated_relations_dict.pkl", "rb") as f:
    annotated_relations = pickle.load(f)


### Matching original text (0, 403) sentences with annotations
### Extracting a list of sentence pairs with the appropriate annotation
all_relations = []

for i in range(1, 403):
    this_text_true_relations = annotated_relations[i]
    with open(fr"Data\AAE\{i}.txt", "r", encoding="utf-8") as f_essay:
        full_essay = f_essay.read()

    doc = nlp(full_essay)

    sentences = [sent.text for sent in doc.sents]

    for j in range(len(sentences)):
        for k in range(j+1, len(sentences)):

            flag = True

            for ann_rel in this_text_true_relations:
                if (
                        (ann_rel[1] in sentences[j] and ann_rel[2] in sentences[k])
                        or
                        (ann_rel[2] in sentences[j] and ann_rel[1] in sentences[k])
                ):
                    all_relations.append([clean_sent(sentences[j]), clean_sent(sentences[k]), ann_rel[0]])
                    flag = False

            if flag:
                all_relations.append([clean_sent(sentences[j]), clean_sent(sentences[k]), 'neutral'])

### Creating a list for each unique annotation
### (1 list for entailments, 1 for contraditions, 1 for neutral)
elist = []
clist = []
nlist = []
for i, line in enumerate(all_relations):
    if line[2] == 'entails':
        elist.append(i)
    elif line[2] == 'contradicts':
        clist.append(i)
    elif line[2] == 'neutral':
        nlist.append(i)


### Choose 300 entailments, 100 contradictions and 600 neutral sentence pairs at random
### The 300-100-600 split is done to represent the distribution of the annotated labels
random_indices = random.sample(elist, 300) + random.sample(clist, 100) + random.sample(nlist, 600)
random.shuffle(random_indices)

sampled_relations = []
for r_ind in random_indices:
    sampled_relations.append(all_relations[r_ind])


##### ##### ##### ##### ##### ##### UNCOMMENT TO STORE ##### ##### ##### ##### ##### #####
### The final part of this file is used to store the randomized 1000 samples

### NOTE   This randomized setting was used to produce the paper results
### NOTE   The 1k random samples that can be found under the Data directory are the original ones
### NOTE   If you wish to reproduce the metrcis and evaluations, please reuse the existing data
### NOTE   The random seed was not set (thus using default value)


# to_save = pd.DataFrame(sampled_relations, columns=['sentence1', 'sentence2', 'label'])
# to_save.to_csv(r"Data\data.csv", encoding='UTF-8', sep=';')

# with open(r"Data\AAE_all_sentences.pkl", 'wb') as handle:
#    pickle.dump(all_relations, handle)
