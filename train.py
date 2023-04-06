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

def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss

def run_evaluation_pipeline(model_name):

  trained_model = tf.keras.models.load_model(model_name, custom_objects={"contrastive_loss": contrastive_loss})
  newModel = Model(inputs=trained_model.input, outputs=trained_model.layers[3].input)
  sampler = RandomOverSampler() #RandomUnderSampler
  
  def evaluate(x,y,model):
      rus = RandomUnderSampler(random_state=42)
      print(x.shape)
      x = x.reshape(x.shape[0], 50, 96)
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

      x_train, y_train = rus.fit_resample(x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]), y_train)
      x_train = x_train.reshape(x_train.shape[0], 50, 96)

      embedding_train = (model.predict([x_train, x_train])[0])
      embedding_test = (model.predict([x_test, x_test])[0])

      clf = GradientBoostingClassifier(n_estimators=20)
      clf.fit(embedding_train, y_train)
        
      score = f1_score(y_test, clf.predict(embedding_test))
      return score

  tasks = ["spam", "toxic", "yelp_reviews", "imdb", "spamemail"]
  print(tasks)
  all_task_score = {}
  for task in tasks:
    task_list_score = []
    for i in range(0,20):
      x=np.load("evaluation_tasks/"+task+"_task_x" + str(i) +".npy")
      y=np.load("evaluation_tasks/"+task+"_task_y" + str(i) +".npy")
      score = evaluate(x,y,newModel)
      task_list_score.append(score)
    
    all_task_score[task] = task_list_score
  return all_task_score

model_path = "models/unsupervised_combined"
all_scores = run_evaluation_pipeline(model_path)
print(model_path)
for s in all_scores:
    mean = np.mean(np.array(all_scores[s]))
    std = np.std(np.array(all_scores[s]))
    print("task: ", s)
    print("f1 mean: ", mean, " std: ", std)
    

