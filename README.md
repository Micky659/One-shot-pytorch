# One-shot-pytorch
Pytorch implementation of one shot learning for face matching. 
The implementation uses siamese network and contrasive loss for training. One shot learning is used where there isn't much training data for the neural net.


## Preprocessing
The preprocessing and support script is hardcoded based on the dataset available to me, you will have to modify the preprocessing.py if you have an unprocessed data of face images.

If you want to use the dataset i used you can download it from this link- [trainset](https://drive.google.com/file/d/12_WTFi9ppvD-loaWUWpUar25Z3nT5k9P/view)

Dataset size- 427MB


## Training 
Extract the dataset in the core project directory and run the preprocessing script and it will preprocess the data required as input, use the support script if the preprocessing halts to get the last preprocessed data location and modify the preprocessing script to resume the task.

Run the train script after this, change the num_workers according to your system specification. The train script will create a model checkpoint in the directory model, create if not already present.


## Testing
Use the test script to test on sample inputs, run the script and provide loaction of two images as input, create a folder testSpace and save the two input images in that directory.

It will use the preprocessed model and output the cosine similarity between the two pictures with the output similar or dissimilar.

<p>&nbsp;</p>

***Feel free to build over my code and use it wisely***
