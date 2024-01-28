# GAI_gpt2
Generation of text in the style of Asimov Isaac using gpt2

# About the data
Since the dataset is composed by epub files of books, the dataset can not be loaded due to copyright.

# About the model
The model checkpoints, being too heavy to be loaded on github, are available [here](https://drive.google.com/drive/folders/1cwLcoa0xdbkVNu3dZ2jMMqjkh9z37rPt?usp=sharing).

# About the code
The code is structured in different python notebooks.

[`data_preprocessing.ipynb`](data_preprocessing.ipynb) is used to create data from a collection of books in epub format, although some manual editing is needed.

[`fine_tune_gai.ipynb`](fine_tune_gai.ipynb) manages the fine-tuning of the gpt2 model on the Asimov dataset.

[`generation.ipynb`](generation.ipynb), along with [`decoding_functions.py`](utils/decoding_functions.py), handles the inference step of the fine-tuned model, in order to generate sentences in the style of Isaac Asimov.

[`asimov_classifier.ipynb`](asimov_classifier.ipynb), finally, evaluate the performances of the GAI model, using a transformer-based classifier (BERT) as metric.
