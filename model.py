from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.title("Training Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	# plt.savefig(plotPath)
	plt.show()

# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

def build_siamese_model_nlp(inputShape, embeddingDim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    # x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.3)(x)
    # # second set of CONV => RELU => POOL => DROPOUT layers
    # x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    # x = MaxPooling2D(pool_size=2)(x)
    # x = Dropout(0.3)(x)
    x = LSTM(128)(inputs)
    #x = LayerNormalization()(x)
    # x = Embedding(64,48)(x)
    x = Dropout(0.3)(x)
    # prepare the final outputs
    outputs = Dense(embeddingDim)(x)
    # build the model
    model = Model(inputs, outputs)
    print(model.summary())
    # return the model to the calling function
    return model

def load_data(dataset_name):
  pairTrainNP_1 = np.load(dataset_name + "_train0.npy")
  pairTrainNP_2 = np.load(dataset_name + "_train1.npy")
  labelTrain = np.load(dataset_name + "_trainLabel.npy")
  return pairTrainNP_1, pairTrainNP_2, labelTrain

def load_all_dataset():
    dataset_name = "yelp_reviewsunsup"
    yelp_pairTrainNP_1, yelp_pairTrainNP_2, yelp_labelTrain = load_data(dataset_name)
    dataset_name = "toxicunsup"
    toxic_pairTrainNP_1, toxic_pairTrainNP_2, toxic_labelTrain = load_data(dataset_name)
    dataset_name = "spamunsup"
    spam_pairTrainNP_1, spam_pairTrainNP_2, spam_labelTrain = load_data(dataset_name)
    dataset_name = "spamemailunsup"
    email_pairTrainNP_1, email_pairTrainNP_2, email_labelTrain = load_data(dataset_name)
    dataset_name = "imdbunsup"
    imdb_pairTrainNP_1, imdb_pairTrainNP_2, imdb_labelTrain = load_data(dataset_name)
    pairTrainNP_1 = np.concatenate([spam_pairTrainNP_1, toxic_pairTrainNP_1, yelp_pairTrainNP_1, email_pairTrainNP_1, imdb_pairTrainNP_1],axis=0)
    pairTrainNP_2 = np.concatenate([spam_pairTrainNP_2, toxic_pairTrainNP_2, yelp_pairTrainNP_2, email_pairTrainNP_2, imdb_pairTrainNP_2],axis=0)
    labelTrain = np.concatenate([spam_labelTrain, toxic_labelTrain, yelp_labelTrain, email_labelTrain, imdb_labelTrain],axis=0)
    return pairTrainNP_1, pairTrainNP_2, labelTrain

def train_model(model_path, pairTrainNP_1, pairTrainNP_2, labelTrain):
    
    print("[INFO] building siamese network for nlp...")
    sentenceA = Input(shape=(pairTrainNP_1.shape[1], 96))
    sentenceB = Input(shape=(pairTrainNP_1.shape[1], 96))
    featureExtractor = build_siamese_model_nlp((pairTrainNP_1.shape[1], 96))
    featsA = featureExtractor(sentenceA)
    featsB = featureExtractor(sentenceB)
    # finally, construct the siamese network
    distance = Lambda(euclidean_distance)([featsA, featsB])
    model = Model(inputs=[sentenceA, sentenceB], outputs=distance)

    # compile the model
    print("[INFO] compiling model...")
    model.compile(loss=contrastive_loss, optimizer="Adam")
    # train the model
    print("[INFO] training model...")
    model.fit(
        [pairTrainNP_1, pairTrainNP_2],  labelTrain,
        #alidation_data = validation_data,
        validation_split = 0.1,
        batch_size=128,
        epochs=10)
    # serialize the model to disk
    print("[INFO] saving siamese model...")
    model.save(model_path)
    # plot the training history
    print("[INFO] plotting training history...")
    #plot_training(history, PLOT_PATH)