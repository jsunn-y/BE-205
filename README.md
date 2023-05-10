# BE-205

To run, install the conda environment from `env.yml`

## Assignment 4 Report

Jupyter notebook demo: `Assignment 4/main.ipynb`

We chose to use a 1D convolutional neural network, following by a single dense layer to learn meaningful embeddings for protein sequences. Namely, we chose to predict the text token from contexts of length 20 amino acids. Even though we had access to a GPU, we limited the training set size signficantly and only trained for 10 epochs. The model seems to converge to the final loss, although it is not very low, and it only does somewhat better than randomly guessing (as calculating by the log likelihood of the cross entropy loss).

### Visualization of Amino Acid Embeddings
<img src="Assignment 4/embeddings.png">

## Assignment 1 Report

Weights saved from the logistic regression model: `Assignment 1/model.sav`

Function to test code (used for grading):  `run_test_code` from `Assignment 1/util.py`

Jupyter notebook demo: `Assignment 1/main.ipynb`

### Marker Expression Panel:
<img src="Assignment 1/marker_expression.jpg">

### Confusion Matrix:
<img src="Assignment 1/confusion_matrix.jpg">

### Analysis: 
We performed multiclass classification of individual cells into one of 18 cell types, based on the mean expression profile across the cell, for 51 different markers. From marker expression panel analysis, some of the markers with the biggest impact include Au; Na; and the sum of H3K27m3, H3K9ac, and dsDNA. Most of the markers have relatively low impact.

We then trained a logisitic regression model to predict the classification of each individual cell, based on the mean expression profile. While this is a very basic model, as it is linear, it still performed decently. For most cell types, the the correct cell type was predicted most often. However, notably, the model struggles to correctly classify endothelial, Nk, and Mono-Neu cells, often getting them confused for other cells. 

We suggest that the performance of the model could be improved by introducing a more expressive model, such as a neural network. Furthermore, regularization could be used to reduce the weights in some of the less important marker features, reducing overfitting.
