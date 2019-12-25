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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Input, Bidirectional, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, model_from_yaml
from keras.layers.core import Reshape
from keras.layers import GlobalMaxPooling1D, Flatten
from sklearn.metrics import classification_report, confusion_matrix
from keras.regularizers import l2

from losses import dice
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
  parser.add_argument("--visible_device_list", default='1',
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
  # best model is already saved
  #model.save_weights( os.path.join(model_dir, "model.h5"))

  print("Model saved to disk: " + model_dir)


def load_model(model_dir):
  # load YAML and create model
  yaml_file = open(os.path.join(model_dir, 'model.yaml'), 'r')
  loaded_model_yaml = yaml_file.read()
  yaml_file.close()
  loaded_model = model_from_yaml(loaded_model_yaml)

  # load weights into new model
  loaded_model.load_weights(os.path.join(model_dir, 'model.h5'))
  print("Loaded model from disk: %s" % (model_dir))

  return loaded_model

def write_results(output_tsv, gs_path, pred, labels_dic):
  """
  Write list of output in official format
  :param output_tsv:
  :param pred:
  :return:
  """

  ft = open(gs_path)
  lines = ft.readlines()
  assert len(lines) == len(pred),  'line inputs does not match: input vs. pred : %d / %d' % (len(lines), len(pred))

  NAs_count = 0
  with open(output_tsv, 'w') as fo:
    for pred_idx, line in zip(pred, lines):
      splits = line.strip().split('\t')
      if labels_dic[pred_idx] == "NA":
        NAs_count = NAs_count + 1
        continue

      fo.write("%s\t%s\tArg1:%s\tArg2:%s\n" % (splits[-1], labels_dic[pred_idx],splits[-3], splits[-2]))

  print("results written: " + output_tsv, 'NAs_count', NAs_count)
  ft.close()

def do_test(model, sequence, positon1_sequence, position2_sequence, y_test, labels_dic):
  """
  Run official submission
  :param model
  :return:
  """

  print('Starting evaluating')

  y_gs = y_test
  pred = model.predict([sequence, positon1_sequence, position2_sequence], verbose=False).argmax(axis=-1)
  print(pred.shape, 'pred.shape ')
  gs_txt = 'data/sentenses/test.txt'

  output_tsv = 'pred_test.tsv'
  gs_tsv = '../../data/chemprot_test/chemprot_test_gold_standard.tsv'

  # official eval has different working directory (./eval)

  write_results(os.path.join('eval', output_tsv), gs_txt, pred, labels_dic)

  os.chdir('eval')
  os.system("sh ./eval.sh %s %s" % (output_tsv, gs_tsv))
  os.chdir('..')

  #print('Confusion Matrix: ')
  #print(confusion_matrix(y_gs, pred))

  #print()
  #print('Classification Report:')
  #print(classification_report(y_gs, pred, labels=list(labels_dic.keys()),target_names=list(labels_dic.values()),digits=5))

def plot_metrices(history):

  acc = history.history["acc"]
  val_acc = history.history["val_acc"]
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  x_range = range(len(acc))

  plt.figure(1)
  plt.title('Traning and Validation Losses')
  plt.xlabel("Ephocs")
  plt.ylabel("Loss")
  plt.plot(x_range, loss, 'b', label='Training loss')
  plt.plot(x_range, val_loss, 'r', label='Validation loss')  #x_range, loss, 'bo', x_range, val_loss, 'r', x_range, val_loss, 'ro')
  plt.legend()
  plt.savefig('1.loss.png')

  plt.figure(2)
  plt.title('Traning and Validation Accuracy')
  plt.xlabel("Ephocs")
  plt.ylabel("Accuracy")
  plt.plot(x_range, acc, 'b', label='Training Accuracy')
  plt.plot(x_range, val_acc, 'r', label='Validation Accuracy')  #x_range, loss, 'bo', x_range, val_loss, 'r', x_range, val_loss, 'ro')
  plt.legend()
  plt.savefig('2.acc.png')


def words_to_posotion(sentences, ref_position1, ref_position2):
  position1_sentences = []
  position2_sentences = []
  i = 0
  for sentence in sentences:
    words = sentence.split(' ')
    position1_sentence = [str(words.index(word)-ref_position1[i]) for word in words]
    position1_sentences.append(' '.join(position1_sentence))

    position2_sentence = [str(words.index(word)-ref_position2[i]) for word in words]
    position2_sentences.append(' '.join(position2_sentence))

    i = i+1

  return position1_sentences, position2_sentences

def main(args):

  colums_names = ['group', 'chem', 'gen', 'loc1', 'loc2', 'sentense', 'arg1', 'arg2', 'pmid']

  train_data=pd.read_csv(args.train_file, sep='\t', lineterminator='\n', header=None, names=colums_names, keep_default_na=False)
  dev_data=pd.read_csv(args.dev_file, sep='\t', lineterminator='\n', header=None, names=colums_names, keep_default_na=False)
  train_data = train_data.append(dev_data, ignore_index=True)

  test_data=pd.read_csv(args.test_file, sep='\t', lineterminator='\n', header=None, names=colums_names, keep_default_na=False)

  # See how the data looks
  #print(train_data.head(1))
  print('shapes training and testing', train_data.shape, test_data.shape)

  # Check for missing data
  #print(train_data.isnull().sum())
  #print(dev_data.isnull().sum())
  #print(test_data.isnull().sum())

  groups=train_data.group.unique()
  groups.sort()
  dic={}
  labels_dic = {}
  for i,group in enumerate(groups):
    dic[group]=i
    labels_dic[i] = group

  train_labels=train_data.group.apply(lambda x:dic[x])
  test_labels=test_data.group.apply(lambda x:dic[x])
  print('labels dictonary', dic)

  sentenses = train_data.sentense

  tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)
  tokenizer.fit_on_texts(sentenses)
  sequences_train = tokenizer.texts_to_sequences(sentenses)
  sequences_test=tokenizer.texts_to_sequences(test_data.sentense)
  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))

  X_train = pad_sequences(sequences_train)
  X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1])

  #Words Positions with repect to first entity
  # num_words=maximum number of positions = maximum sentence length

  position1_sentenses_train, position2_sentenses_train = words_to_posotion(
             train_data.sentense.values.tolist(), train_data.loc1.values.tolist(), train_data.loc2.values.tolist())
  position1_sentenses_test, position2_sentenses_test = words_to_posotion(
             test_data.sentense.values.tolist(), test_data.loc1.values.tolist(), test_data.loc2.values.tolist())

  position_tokenizer = Tokenizer(num_words=X_train.shape[1],filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)
  position_tokenizer.fit_on_texts(position1_sentenses_train)

  position1_sequences_train = position_tokenizer.texts_to_sequences(position1_sentenses_train)
  position1_sequences_test = position_tokenizer.texts_to_sequences(position1_sentenses_test)

  position2_sequences_train = position_tokenizer.texts_to_sequences(position2_sentenses_train)
  position2_sequences_test = position_tokenizer.texts_to_sequences(position2_sentenses_test)

  position_index = position_tokenizer.word_index
  print('Found %s unique position tokens.' % len(position_index))

  X_position1_train = pad_sequences(position1_sequences_train, maxlen=X_train.shape[1])
  X_position2_train = pad_sequences(position2_sequences_train, maxlen=X_train.shape[1])

  X_position1_test = pad_sequences(position1_sequences_test, maxlen=X_train.shape[1])
  X_position2_test = pad_sequences(position2_sequences_test, maxlen=X_train.shape[1])


  y_train = to_categorical(np.asarray(train_labels[train_data.index]))
  y_test = to_categorical(np.asarray(test_labels[test_data.index]))

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

  words_embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)

  # Words
  words_input = Input(shape=(X_train.shape[1],))
  words_out = words_embedding_layer(words_input)
  words_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(words_out)

  # Position1
  position1_input = Input(shape=(X_train.shape[1],))
  position1_out = Embedding(X_train.shape[1], int(EMBEDDING_DIM/2))(position1_input)
  position1_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(position1_out)

  # Position2
  position2_input = Input(shape=(X_train.shape[1],))
  position2_out = Embedding(X_train.shape[1], int(EMBEDDING_DIM/2))(position2_input)
  position2_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(position2_out)

  concat = Concatenate()([words_out, position1_out, position2_out])

  output = GlobalMaxPooling1D()(concat)
  #output = Flatten()(concat)
  #output = Dropout(0.4)(output)
  #output = Dense(128,kernel_regularizer=l2(0.001), activation='relu')(output)
  output = Dense(128, activation='relu')(output)
  output = Dropout(0.2)(output)
  output = Dense(len(dic), activation='softmax')(output)

  model = Model(inputs=[words_input, position1_input, position2_input], outputs=output)

  model.compile(optimizer='adam', loss=dice, metrics=['accuracy'])

  model.summary()

  checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

  history = model.fit([X_train, X_position1_train, X_position2_train], y_train, batch_size=200, epochs=200000,
       verbose=1, #steps_per_epoch = 20,
       validation_data=([X_test, X_position1_test, X_position2_test], y_test), #validation_steps = 15,
       callbacks=[checkpoint, es])

  save_model('./', model)

  model = load_model('./')

  print('Ploting loss and accuracy')
  plot_metrices(history)

  print("Training done. Model saved at: ")

  do_test(model, X_test, X_position1_test, X_position2_test , y_test, labels_dic)


if __name__ == "__main__":

  args = argparser()

  # set GPU parameters
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = args.visible_device_list
  set_session(tf.Session(config=config))

  main(args)
