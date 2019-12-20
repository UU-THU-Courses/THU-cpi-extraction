#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
from modules.utilities.utils import write_to_file_txt
from nltk.corpus import wordnet as wn
import pandas as pd
import re

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global path variables for dataset files                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
train_entity = "data/chemport/chemprot_training/chemprot_training_entities.tsv"
devel_entity = "data/chemport/chemprot_development/chemprot_development_entities.tsv"
tests_entity = "data/chemport/chemprot_test_gs/chemprot_test_entities_gs.tsv"

train_feat = "data/sentenses/train_ftr.txt"
devel_feat = "data/sentenses/test_ftr.txt"
tests_feat = "data/sentenses/development_ftr.txt"

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global path variables for newly created data files.                                  #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
inter_words = "data/others/extracted_interactions.txt"

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Column headers for different types of data files                                            #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
column_names_entities = ['pmid', 'entity-number', 'type', 'start', 'end', 'name']
column_name = ['label', 'chemical', 'gene', 'loc_1', 'loc_2', 'sentence', 'pmid']

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   main function to interaction words from the corpus of document provided.                    #
#                                                                                               #
#***********************************************************************************************#
def get_interaction_words():
    # read data file to extract relations
    train_data = pd.read_csv(train_feat, sep=',', lineterminator='\n', header=None, names=column_name, keep_default_na=False)
    devel_data = pd.read_csv(devel_feat, sep=',', lineterminator='\n', header=None, names=column_name, keep_default_na=False)
    tests_data = pd.read_csv(tests_feat, sep=',', lineterminator='\n', header=None, names=column_name, keep_default_na=False)
    # read the entities data file for further refinement of dictionary
    train_entities = pd.read_csv(train_entity, sep='\t', lineterminator='\n', header=None, names=column_names_entities, keep_default_na=False)
    devel_entities = pd.read_csv(devel_entity, sep='\t', lineterminator='\n', header=None, names=column_names_entities, keep_default_na=False)
    tests_entities = pd.read_csv(tests_entity, sep='\t', lineterminator='\n', header=None, names=column_names_entities, keep_default_na=False)
    # read wordnet corpus to make certain adjustments
    adverbs = {x.name().split('.', 1)[0] for x in wn.all_synsets('r')}
    nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
    adjec = {x.name().split('.', 1)[0] for x in wn.all_synsets('a')}
    adjec_sat = {x.name().split('.', 1)[0] for x in wn.all_synsets('s')}
    # call the extraction function for each dataset
    set_1 = _interaction_words(train_data, train_entities, nouns, adjec, adjec_sat, adverbs)
    set_2 = _interaction_words(devel_data, devel_entities, nouns, adjec, adjec_sat, adverbs)
    set_3 = _interaction_words(tests_data, tests_entities, nouns, adjec, adjec_sat, adverbs)
    # get a union of all three sets
    union_set = set_1.union(set_2, set_3)
    # finally write the final set to the text file
    write_to_file_txt(union_set, inter_words)

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to extract interaction words based on entities data provided.                      #
#                                                                                               #
#***********************************************************************************************#
def _interaction_words(sentence_data, entities_data, wordnet_nouns, wordnet_adjec, wordnet_adjec_sat, wordnet_advrb):
    entity_dict = {entity for entity in entities_data['name'].values.tolist()}
    rel_set = {""}
    # a regex to match the new interaction word with
    regex = re.compile("[a-zA-Z]*$")
    # iterate over the entire sentence data and extract important words
    for entry in sentence_data[['loc_1', 'loc_2', 'sentence']].values.tolist():
        tokens = entry[2].split(" ")
        start = entry[0] if entry[0]<entry[1] else entry[1]
        end = entry[0] if entry[0]>entry[1] else entry[1]
        for index in range(start+1, end):
            if not bool(re.match(regex, tokens[index])):
                continue
            if tokens[index] in wordnet_nouns:
                continue
            if tokens[index] in wordnet_adjec:
                continue
            if tokens[index] in wordnet_adjec_sat:
                continue
            if tokens[index] in wordnet_advrb:
                continue
            if tokens[index] in entity_dict:
                continue
            if len(tokens[index])<4:
                continue
            rel_set.add(tokens[index])
    # return the newly created interaction word set
    return rel_set
