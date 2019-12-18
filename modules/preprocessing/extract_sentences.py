#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
from nltk.tokenize.punkt import PunktSentenceTokenizer
import pandas as pd
import itertools
from modules.utilities.utils import write_to_file_csv, get_word_index

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
train_folder = "data/chemport/chemprot_training/"
devel_folder = "data/chemport/chemprot_development/"
tests_folder = "data/chemport/chemprot_test/"

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global path variables for newly created data files.                                  #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
train_embd = "data/sentenses/train_emb.txt"
devel_embd = "data/sentenses/test_emb.txt"
tests_embd = "data/sentenses/development_emb.txt"

train_feat = "data/sentenses/train_ftr.txt"
devel_feat = "data/sentenses/test_ftr.txt"
tests_feat = "data/sentenses/development_ftr.txt"

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Column headers for different types of data files                                            #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
column_names_entities = ['pmid', 'entity-number', 'type', 'start', 'end', 'name']
column_names_relation = ['pmid', 'cpr-group', 'eval', 'cpr', 'arg_1', 'arg_2']
column_names_abstract = ['pmid', 'title', 'abstract']

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   main function to extract sentences from the dataset of chemport.                            #
#                                                                                               #
#***********************************************************************************************#
def get_sentences():
    # read training dataset files
    train_entities = pd.read_csv(train_folder+"chemprot_training_entities.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_entities,
                                 keep_default_na=False)
    train_relation = pd.read_csv(train_folder+"chemprot_training_relations.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_relation,
                                 keep_default_na=False)
    train_abstract = pd.read_csv(train_folder+"chemprot_training_abstracts.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_abstract,
                                 keep_default_na=False)

    # read development dataset files
    devel_entities = pd.read_csv(devel_folder+"chemprot_development_entities.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_entities,
                                 keep_default_na=False)
    devel_relation = pd.read_csv(devel_folder+"chemprot_development_relations.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_relation,
                                 keep_default_na=False)
    devel_abstract = pd.read_csv(devel_folder+"chemprot_development_abstracts.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_abstract,
                                 keep_default_na=False)

    # read testing dataset files
    tests_entities = pd.read_csv(tests_folder+"chemprot_test_entities.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_entities,
                                 keep_default_na=False)
    tests_abstract = pd.read_csv(tests_folder+"chemprot_test_abstracts.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_abstract,
                                 keep_default_na=False)

    # getting sentences from each of the datasets
    trn_sentences = _sentences(train_abstract, train_entities)
    dev_sentences = _sentences(devel_abstract, devel_entities)
    tst_sentences = _sentences(tests_abstract, tests_entities)

    # write to file so that it can be used later for feature extraction
    write_to_file_csv(trn_sentences, train_feat)
    write_to_file_csv(dev_sentences, devel_feat)
    write_to_file_csv(tst_sentences, tests_feat)

    # create a corpus useable by the embedding algorithms
    create_corpus(trn_sentences, train_embd)
    create_corpus(dev_sentences, devel_embd)
    create_corpus(tst_sentences, tests_embd)

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to extract sentences based on entities data provided.                              #
#                                                                                               #
#***********************************************************************************************#
def _sentences(abstracts_data, entities_data):
    sentence_data = []

    for line in abstracts_data[["pmid", "abstract"]].to_numpy():
        # get unique (entity, type) tuples
        current_entities = list(set(map(tuple,entities_data[entities_data["pmid"]==line.item(0)][["name", "type"]].to_numpy())))
        # sort the list in descending order of length of the names to avoid common string issue
        current_entities.sort(key = lambda s: len(s[0]), reverse=True)
        # parse abstract into sentences
        sentences_untreated = list(punkt_tokenizer.sentences_from_text(line.item(1)))
        # check if each sentence is valid base on it having atleast one entity.
        for sent in sentences_untreated:
            for source, target in itertools.permutations(current_entities, 2):
                # skip if both entities are the same
                if source[1] == target[1]:
                    continue
                if source[1] != "CHEMICAL":
                    continue
                if source[0] in target[0] or target[0] in source[0]:
                    continue
                # check if the sentence has the current entity pair
                if (source[0] in sent) and (target[0] in sent) :
                    loc_1 = get_word_index(source[0], sent)
                    loc_2 = get_word_index(target[0], sent)
                    sentence_data.append([source[0], target[0], loc_1, loc_2, sent, line.item(0)]) # add labels/CPR groups

    # return the newly created list of sentences
    return sentence_data

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to create a corpus of the extracted sentences for later use by other modules.      #
#                                                                                               #
#***********************************************************************************************#
def create_corpus(sentences, file_name="ABCD.txt"):
    pass
