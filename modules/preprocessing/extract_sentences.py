#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
from nltk.tokenize.punkt import PunktSentenceTokenizer
import pandas as pd
import itertools
from modules.utilities.utils import write_to_file_csv, write_to_file_tsv, get_word_index

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
tests_folder = "data/chemport/chemprot_test_gs/"

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
column_names_entities = ['pmid', 'tag', 'type', 'start', 'end', 'name']
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
    tests_entities = pd.read_csv(tests_folder+"chemprot_test_entities_gs.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_entities,
                                 keep_default_na=False)
    tests_relation = pd.read_csv(tests_folder+"chemprot_test_relations_gs.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_relation,
                                 keep_default_na=False)
    tests_abstract = pd.read_csv(tests_folder+"chemprot_test_abstracts_gs.tsv", sep='\t',
                                 lineterminator='\n', header=None, names=column_names_abstract,
                                 keep_default_na=False)
    # getting sentences from each of the datasets
    trn_sent_feature,trn_sent_embedd  = _sentences(train_abstract, train_entities, train_relation)
    dev_sent_feature,dev_sent_embedd = _sentences(devel_abstract, devel_entities, devel_relation)
    tst_sent_feature,tst_sent_embedd = _sentences(tests_abstract, tests_entities, tests_relation)
    # write to file so that it can be used later for feature extraction
    write_to_file_csv(trn_sent_feature, train_feat)
    write_to_file_csv(dev_sent_feature, devel_feat)
    write_to_file_csv(tst_sent_feature, tests_feat)
    # create a corpus useable by the embedding algorithms
    write_to_file_tsv(trn_sent_embedd, train_embd)
    write_to_file_tsv(dev_sent_embedd, devel_embd)
    write_to_file_tsv(tst_sent_embedd, tests_embd)

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to extract sentences based on entities data provided.                              #
#                                                                                               #
#***********************************************************************************************#
def _sentences(abstracts_data, entities_data, relations_data):
    sentence_data_embedding = []
    sentence_data_features = []
    # iterate over all abstracts and extract sentences according to our requirements
    for line in abstracts_data[["pmid", "abstract", "title"]].to_numpy():
        title_len = len(line.item(2))
        # get unique (entity, type) tuples
        current_entities = list(set(map(tuple,entities_data[entities_data["pmid"]==line.item(0)][["name", "type"]].to_numpy())))
        # sort the list in descending order of length of the names to avoid common string issue
        current_entities.sort(key = lambda s: len(s[0]), reverse=True)
        # parse abstract into sentences
        sentences_spans = list(punkt_tokenizer.span_tokenize(line.item(1)))
        sentences_untreated = list(punkt_tokenizer.sentences_from_text(line.item(1)))
        # check if each sentence is valid base on it having atleast one entity.
        for i, sent in enumerate(sentences_untreated):
            sent_start = sentences_spans[i][0]
            sent_end = sentences_spans[i][1]
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
                    tag_1 = entities_data[(entities_data['pmid']==line.item(0)) & (entities_data['name']==source[0]) & (entities_data['start']-title_len>=sent_start) & (entities_data['end']-title_len<=sent_end)][["tag"]].values
                    tag_2 = entities_data[(entities_data['pmid']==line.item(0)) & (entities_data['name']==target[0]) & (entities_data['start']-title_len>=sent_start) & (entities_data['end']-title_len<=sent_end)][["tag"]].values
                    # if either of the tags are empty skip adding this sentence to the corpus
                    if not len(tag_1) or not len(tag_2):
                        continue
                    for tag_pair in itertools.product(tag_1, tag_2):
                        # check relations data to verify if sentence needs to be added to corpus or not
                        cpr_group = relations_data[(relations_data['pmid']==line.item(0)) & (relations_data['arg_1']==("Arg1:"+tag_pair[0][0])) & (relations_data['arg_2']==("Arg2:"+tag_pair[1][0])) & (relations_data['eval']=="Y ")][["cpr-group"]].values
                        if not len(cpr_group):
                            cpr_group = [["NA"]]
                        sentence_data_features.append([cpr_group[0][0], source[0], target[0], loc_1, loc_2, sent, line.item(0)])
                        sentence_data_embedding.append([cpr_group[0][0], source[1], target[1], loc_1, loc_2, sent.replace(source[0],"chemical").replace(target[0],"gene"), tag_pair[0][0], tag_pair[1][0], line.item(0)])

    # return the newly created list of sentences
    return sentence_data_features, sentence_data_embedding
