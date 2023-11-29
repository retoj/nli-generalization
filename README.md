# nli-generalization
Datasets and scripts to accompany the [article](https://doi.org/10.1007/s10849-023-09410-4) in the Journal of Logic, Language, and Information.

Contents of this README:
* [Repository Structure](#repository-structure)
* [Running the all metrics script](#running-the-all-metrics-script)
* [Databases](#databases)
* [Note](#note)
* [Citation](#citation)

## Repository Structure

**Directories**

* `Data/` -> Contains both datasets in the form we used them for evaluations. Each can be found under the folder with the relevant name.

* `Output/` -> Contains all the results from our evaluation runs for Argument Annotated Essays. Files that are in .csv format are considered 'cleaned' while pickle files contain the raw data structures of the evaluation runs.

* `dataset_creating_scripts` -> For using Argument Annotated Essays to evaluate NLI, we had to perform changes to the dataset. In this folder the script we used to scrape the annotations and data and transform them can be found.

* `prediction_models` -> Each python file defines a function that handles the prediction for each model used in this repository (except for chatGPT which has a separate file outside this folder). The chatGPT file defines all the code required to evaluate on chatGPT. Be aware that at this time the results for chatGPT should be different and an API key is needed to connect.

**Files**

* `classify_logical_relationships_mnli.py` -> The scripts for performing and evaluating NLI on the syllogistic dataset.

* `finetune_syll_dgx.ipynb` -> Defines a notebook for fine-tuning models on the syllogistic dataset.

* `metrics.py` -> Contains the script originally used to derive the different results for different evaluation configurations.

* `model_predictions.py` -> Was originally used to manually set a model and perform predictions on the datasets available, storing the output in the `Output` folder under the correct subfolders.

* `print_existing_results.py` -> This was added after the paper publication. It is a script that will print all the evaluation results that we have stored in the `Output` folder. **The output this file prints is only in regards to the Argument Annotated Essays!!**

## Running the all metrics script

Before the `print_existing_results.py` script can be run, some dependencies have to be installed. Before doing anything else, make sure you are in this project directory as shown:

```bash
C:\{your}\{user}\{path}\Capturing the Varieties of Natural Language Inference>
```

### Dependencies

First, the most basic dependencies (those that then install their required dependencies) are all in the `requirements.txt` file. Run

```bash
pip install -r requirements.txt
```

before running any of the scripts to avoid import errors. Furthermore, spacy uses a pre-trained model to tokenize and process natural language which needs to be downloaded as well, running the following command:

```bash
python -m spacy download en
```

Finally, the models themselves are contained in the `prediction_models` folder. Running scripts there will automatically download and cache the model weights that are provided through the Hugging Face interface.

### Running

The pre-defined script that you can run (only requiring pandas and sklearn) is the `print_existing_results.py` file. Run:

```bash
python print_existing_results.py
```

All the rest of the components are connected and can be called in newly defined scripts. It is only important to have the requirements installed before any such actions.

## Databases

*The datasets below can be found under the `Data` folder*.

### Argument Annotated Essays (AAE)

Argument annotated essays are produced [Stab & Gurevych](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) of the Technical University of Darmstadt. Those comprise a set of 403 texts that are annotated for argumentation. In this dataset, each text contains annotations that take the form of parts of full sentences being annotated as parts of argumentative speech.

In our work, we further process this dataset. Parsing all the texts, we create sentence pairs with annotations that resemble NLI datasets (meaning the annotations capture sentence relations and the sentence pair in one line). Unlike the original work, we take the sentence chunks and match them with the full sentence they were derived from. We then match full sentences with their annotations (where they exist) creating a big corpus of sentence pairs with their relation. Out of those, we select 300 random 'entailments', 100 random 'contradictions' and 600 random 'neutral' sentence pairs out of which we create our random 1k sample for AAE. This 300-100-600 split serves to follow the actual label distribution.

#### Citation

```
Stab, Christian; Gurevych, Iryna. Argument Annotated Essays (version 2). (2017). License description. Argument Mining, 409-06 Informationssysteme, Prozess- und Wissensmanagement, 004. Technical University of Darmstadt. https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422
```

### Syllogistic Dataset

TODO

## Note

Evaluation was an iterative process, thus we didn't consider developing a system for evaluations. Regardless, we provide all scripts, model weights and data required for evaluations in this repository, along with the results of our actual runs.

That is also to say, that due to variability in the different labeling schemes, the scripts vary accordingly. But interfaces to the models (except chatGPT which requires an account) are all provided under the `prediction_models` folder.

## Citation

```
Gubelmann, R., Katis, I., Niklaus, C. et al. Capturing the Varieties of Natural Language Inference: A Systematic Survey of Existing Datasets and Two Novel Benchmarks. J of Log Lang and Inf (2023). https://doi.org/10.1007/s10849-023-09410-4
```
