# NLG AI: Natural Language Generation in the style of Asimov Isaac 
Generative language models like the GPT systems by OpenAI have taken the attention of the masses all over the world. Following the trend, this project focuses on analysing if is possible to use this tool, in particular the open source model GPT2 available on HuggingFace, to generate text in the personal writing style of the famous sci-fi author Isaac Asimov.

![gpt-2-output](https://github.com/LucaCorvitto/GAI_gpt2/assets/87773518/0acbc5fd-3364-42aa-a38c-8fbd9e1419a1)

## Output comparison
| GPT-2 |  GAI  | eGAI  |
| ----- | ----  | ----- |
| Alan woke up the next morning and found her in a car with her brother and three other friends. "I was absolutely horrified and it’s just really upsetting," she said. "It’s a huge shame because I’ve had a great life. "I’m going to be on a mission to help those kids." | Alan woke up to find himself face to face with a wall of roboticized debris. "I don’t know what you are thinking, Mr. R. Daneel," he said, "but you know what you are thinking. The other side of the planet is still being habitable. You know it. |  Alan woke up with a vague feeling of euphoria. The First Minister was, of course, a bachelor. His hair was dark and thin, his eyes were fixed, his face uncertain, his eyes still looking through the window. He was wearing a plain white suit, a dark red shirt, a dark gray trousers, a dark blue shirt, a dark gray tie. |

Here it is a comparison between the outputs of the 3 different models tested. The GPT-2 model is the plain gpt2-small from Huggingface without fine-tuning.

The GAI model is the gpt2 model fine-tuned on a dataset composed by the sci-fi novels of Asimov. As it can be seen the model take full hands from the stories, citing also one of the main characters from the Robot Series.

The eGAI model is the gpt2 model fine-tuned on an enlarged version of the previous dataset, enriched with other books of the same author, such as crime novels and science essays. The goal was to analyse if the model could detect underlying patterns and nuances of the specific writing style without focusing on the content. Seeing the output, the model loses its focus on the sci-fi content.

More information about the project and its results are presented in the [`report`](report.pdf).

## About the data
Since the dataset is composed by epub files of books, the dataset can not be loaded due to copyright.

## About the model
The model checkpoints, being too heavy to be loaded on github, are available [here](https://drive.google.com/drive/folders/1cwLcoa0xdbkVNu3dZ2jMMqjkh9z37rPt?usp=sharing).

## About the code
The code is structured in different python notebooks.

[`data_preprocessing.ipynb`](data_preprocessing.ipynb) is used to create data from a collection of books in epub format, although some manual editing is needed.

[`fine_tune_gai.ipynb`](fine_tune_gai.ipynb) manages the fine-tuning of the gpt2 model on the Asimov dataset.

[`generation.ipynb`](generation.ipynb), along with [`decoding_functions.py`](utils/decoding_functions.py), handles the inference step of the fine-tuned model, in order to generate sentences in the style of Isaac Asimov.

[`asimov_classifier.ipynb`](asimov_classifier.ipynb), finally, evaluate the performances of the GAI models, using metrics such as accuracy, recall, precision and F1-score trhough a transformer-based classifier (BERT).
