#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import pandas as pd
from modules.utilities.utils import write_to_file_csv

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global path variables for dataset files                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
train_feat = "data/sentenses/train_ftr.txt"
devel_feat = "data/sentenses/test_ftr.txt"
tests_feat = "data/sentenses/development_ftr.txt"

inter_words_1 = "data/others/extracted_interactions.txt"
inter_words_2 = "data/others/interaction.txt"

train_ft = "data/features/training_features.csv"
devel_ft = "data/features/development_features.csv"
tests_ft = "data/features/testing_features.csv"

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Column headers for different types of data files                                            #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
column_name = ['label', 'chemical', 'gene', 'loc_1', 'loc_2', 'sentence', 'pmid']

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   main function of the module to extract features from the sentences data.                    #
#                                                                                               #
#***********************************************************************************************#
def get_features():
    # read data file to extract relations
    train_data = pd.read_csv(train_feat, sep=',', lineterminator='\n', header=None, names=column_name, keep_default_na=False)
    devel_data = pd.read_csv(devel_feat, sep=',', lineterminator='\n', header=None, names=column_name, keep_default_na=False)
    tests_data = pd.read_csv(tests_feat, sep=',', lineterminator='\n', header=None, names=column_name, keep_default_na=False)
    # read interaction word lists
    inter_1 = pd.read_csv(inter_words_1, sep=',', lineterminator='\n', header=None, names=["interactions"], keep_default_na=False)
    inter_2 = pd.read_csv(inter_words_2, sep=',', lineterminator='\n', header=None, names=["interactions"], keep_default_na=False)
    # get combined dataset
    interact_words = {word for word in inter_1["interactions"].values.tolist()}.union(
                     {word for word in inter_2["interactions"].values.tolist()})
    # get the features
    train_features = _features(train_data, interact_words)
    tests_features = _features(devel_data, interact_words)
    devel_features = _features(tests_data, interact_words)
    # save features for future usage
    write_to_file_csv(train_features, train_ft)
    write_to_file_csv(tests_features, devel_ft)
    write_to_file_csv(devel_features, tests_ft)

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to extract features based on dataset provided.                                     #
#                                                                                               #
#***********************************************************************************************#
def _features(sentence_data, inter_data):
    # features will be [structure, chemical-location, gene-location, interaction-location, chemical-length, gene-length, sentence-length,
    #                   chemical-gene-distance, chemical-interaction-distance, gene-interaction-distance, sparse-vection(all-interactions or topX-interactions)]
    #['label', 'chemical', 'gene', 'loc_1', 'loc_2', 'sentence', 'pmid']
    features = []
    for entry in sentence_data[['label', 'chemical', 'gene', 'loc_1', 'loc_2', 'sentence', 'pmid']].values.tolist():
        sentence = entry[5]
        tokens = sentence.split(" ")
        label = entry[0]
        chem_loc = entry[3]
        gene_loc = entry[4]
        chem_len = len(entry[1].split(" "))
        gene_len = len(entry[2].split(" "))
        sent_len = len(tokens)
        # pass over each token to check if it is an interaction word or not
        for i,token in enumerate(tokens):
            structure = -1
            if not token in inter_data:
                continue
            if chem_loc<i and i<gene_loc:
                structure = 0
            elif gene_loc<i and i<chem_loc:
                structure = 1
            elif chem_loc<gene_loc and gene_loc<i:
                structure = 2
            elif gene_loc<chem_loc and chem_loc<i:
                structure = 3
            elif i<chem_loc and chem_loc<gene_loc:
                structure = 4
            elif i<gene_loc and gene_loc<chem_loc:
                structure = 5
            # append features
            features.append([structure, chem_loc, gene_loc, i, chem_len, gene_len, sent_len, gene_loc-chem_loc, i-chem_loc, gene_loc-i, label])
    # return the newly created features
    return features
