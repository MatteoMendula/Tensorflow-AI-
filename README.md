# TensorFlowAI
TensorFlow project for AI with image classification and word embedding combined.

## Live DEMO on YouTube
https://www.youtube.com/watch?v=oLLSc-lHAaE&t=

## Setup
To use this project you need all required python modules. So to do that run the following command (that will also permit to execute the training and a default program):

```
setup.bat
```

## How to run
To run the training for the image classifier you have to use this command (it may take a while):

```
python scripts/retrain.py --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --image_dir=tf_files/photos
```

To run the program you need to use this command from the main folder:

```
python scripts/main.py
```

You can add also the following options:
```
  -h, --help           show this help message and exit
  --image IMAGE        image to be processed
  --text TEXT          text used for embedding
  --result RESULT      file where to store results
  --training TRAINING  is training needed? (Y or N)
  --num_words NUM_WORDS    number of near words to search
```

## Built With
* [tensorflow-for-poets-2](https://github.com/googlecodelabs/tensorflow-for-poets-2) - Image Classification
* [python_tutorial](https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb) - Word Embedding + cosine similarity instead of euclidean similarity
## Authors
* **Matteo Mendula** (https://gitlab.com/MatteoMendula)
* **Riccardo Salicini** (https://gitlab.com/RiccardoSalicini)
