import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
import numpy as np

aug_synonym = naw.SynonymAug(aug_src='wordnet')
aug_swap = naw.RandomWordAug(action="swap")
aug_delete = naw.RandomWordAug(action='crop')

# sentence augmentation
def augment_sentence(sentence_string):
  choice = np.random.uniform(0,1,1)
  if choice < 0.33:
    return aug_swap.augment(aug_synonym.augment(sentence_string)[0])[0]
  if choice < 0.66:
    return aug_synonym.augment(aug_swap.augment(sentence_string)[0])[0]
  if choice < 1.1:
    if len(sentence_string.split()) < 5:
      return sentence_string
    return aug_swap.augment(aug_delete.augment(sentence_string)[0])[0]
  return 0

# make pairs of data
def make_pairs_nlp(sentences, labels, number_of_datapoints_needed):
  labels_tmp = [x[1] for x in zip(sentences,labels) if len(str(x[0]).split()) < 50 and len(str(x[0]).split()) > 5]
  sentences_tmp = [x[0] for x in zip(sentences,labels) if len(str(x[0]).split()) < 50 and len(str(x[0]).split()) > 5]
  labels = np.array(labels_tmp)
  sentences = np.array(sentences_tmp)
  # initialize two empty lists to hold the (image, image) pairs and
  # labels to indicate if a pair is positive or negative
  pairSentences = []
  pairLabels = []
  # calculate the total number of classes present in the dataset
  # and then build a list of indexes for each class label that
  # provides the indexes for all examples with a given label
  numClasses = len(np.unique(labels))
  idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

  # loop over all images
  print("number of datapoints: ", number_of_datapoints_needed)
  for idxA in range(min(len(sentences), number_of_datapoints_needed)):
    # grab the current image and label belonging to the current
    # iteration
    currentImage = sentences[idxA]
    label = labels[idxA]
    # randomly pick an image that belongs to the *same* class
    # label
    idxB = np.random.choice(idx[label])
    posImage = sentences[idxB]
    # grab the indices for each of the class labels *not* equal to
    # the current label and randomly pick an image corresponding
    # to a label *not* equal to the current label
    negIdx = np.where(labels != label)[0]
    negImage = sentences[np.random.choice(negIdx)]
    # prepare a positive pair and update the images and labels
    # lists, respectively
    currentEmbedding = nlp(str(currentImage))
    posEmbedding = nlp(str(posImage))
    negEmbedding = nlp(str(negImage))
    if len(currentEmbedding) > 50 or len(posEmbedding) > 50 or len(negEmbedding) > 50:
      continue

    pairSentences.append([[x.vector for x in currentEmbedding], [x.vector for x in posEmbedding]])
    pairLabels.append([1])

    # prepare a negative pair of images and update our lists
    #pairSentences.append([[[x] for x in nlp(currentImage).vector], [[x] for x in nlp(negImage).vector]])
    pairSentences.append([[x.vector for x in currentEmbedding], [x.vector for x in negEmbedding]])
    pairLabels.append([0])
    
  # return a 2-tuple of our image pairs and labels
  return (np.array(pairSentences, dtype=object), np.array(pairLabels))

def make_pairs_unsupervised(sentences, datapt):
  sentences_tmp = [' '.join(x.split()[:50]) for x in sentences if len(x.split()) > 5]
  sentences = np.array(sentences_tmp)
  # initialize two empty lists to hold the (image, image) pairs and
  # labels to indicate if a pair is positive or negative
  pairSentences = []
  pairLabels = []
  
  # loop over all images
  print("number of datapoints: ", datapt)
  for idxA in range(min(len(sentences), datapt)):
    if idxA % 1000 == 0:
        print("looping: ", idxA)
    # grab the current image and label belonging to the current
    # iteration
    currentImage = sentences[np.random.randint(0,len(sentences))]
    posImage = augment_sentence(currentImage)
    # grab the indices for each of the class labels *not* equal to
    # the current label and randomly pick an image corresponding
    # to a label *not* equal to the current label
    negImage = sentences[np.random.choice(len(sentences))]
    # prepare a positive pair and update the images and labels
    # lists, respectively
    currentEmbedding = nlp(str(currentImage))[:50]
    posEmbedding = nlp(str(posImage))[:50]
    negEmbedding = nlp(str(negImage))[:50]

    if len(currentEmbedding) > 50 or len(posEmbedding) > 50 or len(negEmbedding) > 50:
      continue
    pairSentences.append([[x.vector for x in currentEmbedding], [x.vector for x in posEmbedding]])
    pairLabels.append([1])

    # prepare a negative pair of images and update our lists
    #pairSentences.append([[[x] for x in nlp(currentImage).vector], [[x] for x in nlp(negImage).vector]])
    pairSentences.append([[x.vector for x in currentEmbedding], [x.vector for x in negEmbedding]])
    pairLabels.append([0])
    
  # return a 2-tuple of our image pairs and labels
  return (np.array(pairSentences, dtype=object), np.array(pairLabels))

def make_pairs_unsupervised_task_similarity(sentences, labels, task_similiarity, data_pt_needed):
  # task_similiarity = matrix
  assert len(sentences) == len(labels)
  assert len(labels) == len(task_similiarity)

  # initialize two empty lists to hold the (image, image) pairs and
  # labels to indicate if a pair is positive or negative
  pairSentences = []
  pairLabels = []
  
  # form a label index set for ease of sampling
  s_positive = []
  s_negative = []
  for idx, l in enumerate(labels):
    s_positive.append(sentences[idx][np.where(l == 1)])
    s_negative.append(sentences[idx][np.where(l == 0)])

  print("number of datapoints per data-set: ", data_pt_needed)
  for idxA in range(data_pt_needed):
    if idxA % 100 == 0:
        print(idxA)
    for dataset_idx in range(min(len(sentences),data_pt_needed)):
      take_from_label = np.random.choice(2, p=[0.1, 0.9])
      
      # take labeled data:
      if take_from_label == 1:
        idx = np.random.randint(0,len(sentences[dataset_idx]))
        currentImage = sentences[dataset_idx][idx]
        currentLabel = labels[dataset_idx][idx]
        # positive augmentation with sampling by making use of task_similarity
        # high probability -> take from task with close similarity
        prob_vector = task_similiarity[dataset_idx]
        positive_choice = np.random.choice(len(prob_vector), p=prob_vector)

        # choose a datapoint from positive_choice data-set with same label and different label as currentLabel
        if currentLabel == 1:
          posImage = np.random.choice(s_positive[positive_choice])
          negImage = np.random.choice(s_negative[positive_choice])
        else:
          posImage = np.random.choice(s_negative[positive_choice])
          negImage = np.random.choice(s_positive[positive_choice])

        # prepare a positive pair and update the images and labels
        # lists, respectively
        currentEmbedding = nlp(str(currentImage))[:50]
        posEmbedding = nlp(str(posImage))[:50]
        negEmbedding = nlp(str(negImage))[:50]
        if len(currentEmbedding) > 50 or len(posEmbedding) > 50 or len(negEmbedding) > 50:
          continue
        pairSentences.append([[x.vector for x in currentEmbedding], [x.vector for x in posEmbedding]])
        pairLabels.append([1])

        # prepare a negative pair of images and update our lists
        #pairSentences.append([[[x] for x in nlp(currentImage).vector], [[x] for x in nlp(negImage).vector]])
        pairSentences.append([[x.vector for x in currentEmbedding], [x.vector for x in negEmbedding]])
        pairLabels.append([0])

      else: # take non-labeled data (use data-augmentation)
        currentImage = sentences[dataset_idx][np.random.randint(0,len(sentences[dataset_idx]))]
        posImage = augment_sentence(currentImage)
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negImage = sentences[dataset_idx][np.random.randint(0,len(sentences[dataset_idx]))]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        currentEmbedding = nlp(str(currentImage))
        posEmbedding = nlp(str(posImage))
        negEmbedding = nlp(str(negImage))
        if len(currentEmbedding) > 50 or len(posEmbedding) > 50 or len(negEmbedding) > 50:
          continue
        pairSentences.append([[x.vector for x in currentEmbedding], [x.vector for x in posEmbedding]])
        pairLabels.append([1])

        # prepare a negative pair of images and update our lists
        #pairSentences.append([[[x] for x in nlp(currentImage).vector], [[x] for x in nlp(negImage).vector]])
        pairSentences.append([[x.vector for x in currentEmbedding], [x.vector for x in negEmbedding]])
        pairLabels.append([0])
    
  # return a 2-tuple of our image pairs and labels
  return (np.array(pairSentences, dtype=object), np.array(pairLabels))