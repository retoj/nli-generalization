"""
The following script is used to load data, iterate through them, get predictions
through the openai api and store them in raw and clean formats.
"""


import os
import openai
import pandas as pd
import pickle

### Input your openai api key in the environments variables as "OPENAI_API_KEY" or change the below call as needed
openai.api_key = os.getenv("OPENAI_API_KEY")


### openai API call function
def request_label(prompt):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    # Extract the message content from the response
    return response["choices"][0]["message"]["content"]


### Process sentence pairs into prompts
def process_df(relations_df, index):

    result = []

    for i in range(index, relations_df.shape[0]):
        prompt = f'In one word, is the relation between the sentences "{relations_df.iloc[i]["sentence1"]}" and "{relations_df.iloc[i]["sentence2"]}" an entailment, contradiction or neutral?'

        pred_label = request_label(prompt)
        result.append(pred_label)
        print(pred_label)

        with open(r"Output\counter.txt", 'w') as f_c:
            f_c.write(str(i))

    return result



##### ##### GET THE DATA ##### #####
o_data = pd.read_csv(
    r"Data\Argument_annotated_essays\AAE_1k_sample.csv",
    encoding='UTF-8', sep=';',
    names=["id", "sentence1", "sentence2", "label"]
)

data = o_data.copy()

##### ##### KEEP INDEX OF DATA ##### #####
### NOTE   Sometimes chat-GPT will refuse and close the connection
### NOTE   This piece of code can be used to log up to which
### NOTE   sentence pair processing has been done, so that
### NOTE   in case of a connection closure, the processing
### NOTE   can continue from the last unprocessed index

with open(r"Output\chatGPT\counter.txt", 'r') as f_cin:
    index = int(f_cin.read())


##### ##### PREDICT ##### #####
predictions = process_df(data, index)

### Dump raw predictions in pickle file for future processing
with open(r'Output\chatGPT\AAE_raw_predictions.pkl', 'wb') as handle:
    pickle.dump(predictions, handle)

### Add predictions to the dictionary structure
data["predicted_relation"] = predictions

### Save clean csv file with sentence pairs, true and predicted relations
data.to_csv(
    r"Output\chatGPT\AAE_1k_sample_with_predictions.csv",
    encoding='UTF-8',
    sep=';'
)


### Testing connections to chat-gpt

# print(data.head())

# prompt = f'In one word, is the relation between the sentences "{data.iloc[0]["sentence1"]}" and "{data.iloc[0]["sentence2"]}" an entailment, contradiction or neutral?'

# print(request_label(prompt))
