#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
from nltk.tokenize.punkt import PunktSentenceTokenizer
import numpy as np
import pandas as pd
import itertools
import tensorflow as tf
import difflib
import re

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used within this program                                     #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
punkt_tokenizer = PunktSentenceTokenizer()
punkt_tokenizer._params.abbrev_types.add('dr')

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global path variables for dataset files                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
train_folder = "/media/usama/Personal/Study/PHD/HTCB/project/data/chemprot_training/"
devel_folder = "/media/usama/Personal/Study/PHD/HTCB/project/data/chemprot_development/"
tests_folder = "/media/usama/Personal/Study/PHD/HTCB/project/data/chemprot_test/"