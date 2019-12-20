#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import csv

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   getting index of a word in a sentence. (word count)                                         #
#                                                                                               #
#***********************************************************************************************#
def get_word_index(word, string):
    #print(word+"    "+string+"\n")
    index = string.find(word)
    location = 0
    for i in range(index):
        if string[i] == ' ':
            location += 1
    if index == -1:
        location = -1
    # return the position of the word in the string (return -1 if not found)
    return location

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to write lists data to a file.                                                     #
#                                                                                               #
#***********************************************************************************************#
def write_to_file_txt(content, path):
    # use csv library to write to the file
    with open(path, "w") as f:
        for line in content:
            f.write(line+"\n")

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to write lists data to a file.                                                     #
#                                                                                               #
#***********************************************************************************************#
def write_to_file_csv(content, path):
    # use csv library to write to the file
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(content)

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to write lists data to a file.                                                     #
#                                                                                               #
#***********************************************************************************************#
def write_to_file_tsv(content, path):
    with open(path, 'w') as f:
        for line in content:
            for i, item in enumerate(line):
                f.write(str(item))
                if i<8:
                    f.write("\t")
            f.write("\n")