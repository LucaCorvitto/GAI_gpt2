# GAI_gpt2
Generation of text in the style of Asimov Isaac using gpt2

# About the code
The code is structured in different python notebooks.

`data_preprocessing.ipynb` is used to create data from a collection of books in epub format, although some manual editing is needed.

`fine_tune_gai.ipynb` manages the fine-tuning of the gpt2 model on the Asimov dataset.

`generation.ipynb`, along with `decoding_functions.py`, handles the inference step of the fine-tuned model, in order to generate sentences in the style of Isaac Asimov.

`asimov_classifier.ipynb`, finally, evaluate the performances of the GAI model, using a transformer-based classifier (BERT) as metric.
