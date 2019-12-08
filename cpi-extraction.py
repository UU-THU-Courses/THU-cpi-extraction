#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import gensim
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.models import Model, model_from_yaml
from keras.layers.core import Reshape
from keras.layers import GlobalMaxPooling1D

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used through out the program                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#

NUM_WORDS=20000
EMBEDDING_DIM=200

# This method is used to parse command line input arguments

def argparser():
  # command line argments
  parser = argparse.ArgumentParser(
            description="CPI arguments parser")
  parser.add_argument("--visible_device_list", default='0',
            help="To specify shich GPUs to use")
  parser.add_argument("--train_file", default='./data/sentenses/training.txt',
            help="File with training sentenses")
  parser.add_argument("--test_file", default='./data/sentenses/test.txt',
            help="File with testing sentenses")
  parser.add_argument("--dev_file", default='./data/sentenses/development.txt',
            help="File with development sentenses")
  parser.add_argument("--word_embedding_model", default='./data/embeddings/PubMed-w2v.bin',
            help="Pretrained word2vec model")
  args = parser.parse_args()

  return args

def save_model(model_dir, model):
  # serialize model to YAML
  model_yaml = model.to_yaml()
  with open(os.path.join(model_dir, "model.yaml"), "w") as yaml_file:
    yaml_file.write(model_yaml)
  # serialize weights to HDF5
  model.save_weights( os.path.join(model_dir, "model.h5"))

  print("Model saved to disk: " + model_dir)


def load_model(model_dir):
  # load YAML and create model
  yaml_file = open(os.path.join(model_dir, 'model.yaml'), 'r')
  loaded_model_yaml = yaml_file.read()
  yaml_file.close()
  loaded_model = model_from_yaml(loaded_model_yaml, custom_objects={'AttentionWithContext': AttentionWithContext})

  # load weights into new model
  loaded_model.load_weights(os.path.join(model_dir, 'model.h5'))
  print("Loaded model %s from disk: %s" % (model_name, model_dir))

  return loaded_model


def main(args):

  colums_names = ['group', 'chem', 'gen', 'loc1', 'loc2', 'sentense', 'arg1', 'arg2', 'pmid']

  train_data=pd.read_csv(args.train_file, sep='\t', lineterminator='\n', header=None, names=colums_names, keep_default_na=False)
  dev_data=pd.read_csv(args.dev_file, sep='\t', lineterminator='\n', header=None, names=colums_names, keep_default_na=False)
  test_data=pd.read_csv(args.test_file, sep='\t', lineterminator='\n', header=None, names=colums_names, keep_default_na=False)

  # See how the data looks
  print(train_data.head(1))
  print('shapes', train_data.shape, dev_data.shape, test_data.shape)

  # CHeck for missing data
  print(train_data.isnull().sum())
  print(dev_data.isnull().sum())
  print(test_data.isnull().sum())

  groups=train_data.group.unique()
  dic={}
  for i,group in enumerate(groups):
    dic[group]=i

  labels=train_data.group.apply(lambda x:dic[x])
  print('labels dictonary', dic)

  sentenses = train_data.sentense

  tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)
  tokenizer.fit_on_texts(sentenses)
  sequences_train = tokenizer.texts_to_sequences(sentenses)
  sequences_dev=tokenizer.texts_to_sequences(dev_data.sentense)
  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))

  X_train = pad_sequences(sequences_train)
  X_dev = pad_sequences(sequences_dev, maxlen=X_train.shape[1])
  y_train = to_categorical(np.asarray(labels[train_data.index]))
  y_dev = to_categorical(np.asarray(labels[dev_data.index]))
  print('Shape of X train and X devlopment tensor:', X_train.shape,X_dev.shape)
  print('Shape of label train and development tensor:', y_train.shape,y_dev.shape)

  word_vectors = Word2Vec.load_word2vec_format(args.word_embedding_model, binary=True)

  vocabulary_size=min(len(word_index)+1,NUM_WORDS)
  embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
  for word, i in word_index.items():
    if i>=NUM_WORDS:
      continue
    try:
      embedding_vector = word_vectors[word]
      embedding_matrix[i] = embedding_vector
    except KeyError:
      embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

  del(word_vectors)


  # A test model

  embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)

  input = Input(shape=(X_train.shape[1],))
  embedding = embedding_layer(input)
  output = LSTM(128, return_sequences=True, dropout=0.5)(embedding)
  output = GlobalMaxPooling1D()(output)
  output = Dense(128, activation='relu')(output)
  output = Dropout(0.2)(output)
  output = Dense(len(dic), activation='softmax')(output)

  model = Model(inputs=input, outputs=output)

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.summary()

  callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

  history = model.fit(X_train, y_train, batch_size=1000, epochs=2, verbose=1, validation_data=(X_dev, y_dev),
         callbacks=callbacks)

  save_model('./', model)

  print("Training done. Model saved at: ")

if __name__ == "__main__":

  args = argparser()

  # set GPU parameters
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = args.visible_device_list
  set_session(tf.Session(config=config))

  main(args)
