import numpy as np
import sys
import argparse
import os
import json
import re
import csv

slang_1000311281 = ["smh", "fwb", "lmfao", "lmao", "lms", "tbh", "rofl", "wtf", "bff", "wyd", "lylc", "brb", "atm", "imao",
         "sml", "btw", "bw",
         "imho", "fyi", "ppl", "sob", "ttyl", "imo", "ltr", "thx", "kk", "omg", "omfg", "ttys", "afn", "bbs", "cya",
         "ez", "f2f",
         "gtr", "ic", "jk", "k", "ly", "ya", "nm", "np", "plz", "ru", "so", "tc", "tmi", "ym", "ur", "u", "sol",
         "fml"]
BristolNorms_1000311281 = {}
Warringer_1000311281 = {}

reader_1000311281 = csv.reader(open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'))
for line in reader_1000311281:
    BristolNorms_1000311281[line[1]] = [line[3], line[4], line[5]]
    # print(BristolNorms_GilhoolyLogie[line[1]])q

reader_1000311281 = csv.reader(open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'))
for line in reader_1000311281:
    Warringer_1000311281[line[1]] = [line[2], line[5], line[8]]


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros((173))

    # print(comment)

    tokens = re.findall(r"[\S]+", comment)
    raw_comment = " "  # comment without tags
    tags = []
    raw_comment_list = []
    for token in tokens:
        if token != '':
            word = token.split('/')
            tags.append(word[1])
            raw_comment += (word[0] + ' ')
            raw_comment_list.append(word[0])

    # print(raw_comment)
    # 1. Number of first-person pronouns
    result = re.findall(r"\b(i|me|my|mine|we|us|our|ours)\b", raw_comment)
    feats[0] = len(result)
    # print("1st person " + str(feats[0]))

    # 2. Number of second-person pronoun
    result = re.findall(r"\b(you|your|yours|u|ur|urs)\b", raw_comment)
    feats[1] = len(result)
    # print("2nd person " + str(feats[1]))

    # 3. Number of third-person pronouns
    result = re.findall(r"\b(he|him|his|she|her|hers|it|its|they|them|their|theirs)\b", raw_comment)
    feats[2] = len(result)
    # print("3rd person " + str(feats[2]))

    # 4. Number of coordinating conjunctions
    feats[3] = tags.count("CC")
    # print("CC " + str(feats[3]))

    # 5. Number of past - tense verbs
    feats[4] = tags.count("VBD")
    # if feats[4] != 0:
    #     print(comment)
    #     print("VBD " + str(feats[4]))

    # 6. Number of future - tense verbs
    result_1 = re.findall(r"\b(\'ll|will|gonna)\b", raw_comment)
    result_2 = re.findall(r"go\/[\w]+\sto\/[\w]+\s[\w]+\/VB", comment)  # ??????????????????????????
    feats[5] = len(result_1) + len(result_2)
    # if len(result_2) != 0:
    #     print(comment)
    #     print("future " + str(feats[5]))
    #     print("\n\n\n")

    # 7. Number of commas
    # result = re.findall(r",/,", comment)
    feats[6] = raw_comment_list.count(",")
    # if len(result) != 0:
    #     print(comment)
    #     print("comma " +  str(feats[6]))
    #     print("\n\n")

    # 8. Number of multi - character punctuation tokens
    result = re.findall(r"[!\"#$%&\(\)*+,-./:;<=>?@[\]^_`{|}~\\][!\"#$%&\(\)*+,-./:;<=>?@[\]^_`{|}~\\]+", raw_comment)
    feats[7] = len(result)
    # if len(result) != 0:
    #     print(raw_comment)
    #     print("multi char pun " +  str(feats[7]))
    #     print("\n\n")

    # 9. Number of common nouns
    feats[8] = tags.count("NN") + tags.count("NNS")
    # print("NN NNS " + str(feats[8]))

    # 10. Number of proper nouns
    feats[9] = tags.count("NNP") + tags.count("NNPS")
    # print("NNP NNPS " + str(feats[9]))

    # 11. Number of adverbs
    feats[10] = tags.count("RB") + tags.count("RBR") + tags.count("RBS")
    # print("RB RBR RBS " + str(feats[10]))

    # 12. Number of wh - words
    feats[11] = tags.count("WDT") + tags.count("WP") + tags.count("WP$") + tags.count("WRB")

    # 13. Number of slang acronyms
    for word in raw_comment_list:
        if word in slang_1000311281:
            feats[12] += 1
    # if feats[12] != 0:
    #     print(comment)
    #     print("slang " + str(feats[12]) +  "\n")

    # 14.Number of words in uppercase (≥3 letters long)
    upperword = 0
    for word in raw_comment_list:
        if re.match(r"[A-Z][A-Z][A-Z]+", word) and len(word) >= 3:
            upperword += 1
    feats[13] = upperword  # should be 0, changed to lowercase in preprocessing

    # 15.Average length of sentences, in tokens
    num_sentence = [x for x in comment.split("\n") if x is not '']
    if len(num_sentence) == 0:
        feats[14] = len(raw_comment_list)
    else:
        feats[14] = len(raw_comment_list) / len(num_sentence)

    # print("tot len "+ str(len(raw_comment_list)) + " sentence " + str(len(num_sentence))+ "\n\n")

    # 16.Average length of tokens, excluding punctuation - only tokens, in characters
    num_token = 0
    total_token_len = 0
    for word in raw_comment_list:
        if re.findall(r"\w+", word):
            total_token_len += len(word)
            num_token += 1
    if num_token != 0:
        feats[15] = total_token_len / num_token

    # 17.Number of sentences
    num_sentence = [x for x in comment.split("\n") if x is not '']
    if len(num_sentence) == 0:
        feats[16] = 1
    else:
        feats[16] = len(num_sentence)

    # 18.Average of AoA(100 - 700) from Bristol, Gilhooly, and Logie norms
    # 19.Average of IMG from Bristol, Gilhooly, and Logie norms
    # 20.Average of FAM from Bristol, Gilhooly, and Logie norms
    # 21.Standard deviation of AoA(100 - 700) from Bristol, Gilhooly, and Logie norms
    # 22.Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    # 23.Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    AoA = []
    IMG = []
    FAM = []
    num = 0
    for word in raw_comment_list:
        if word in BristolNorms_1000311281.keys():
            AoA.append(int(BristolNorms_1000311281[word][0]))
            IMG.append(int(BristolNorms_1000311281[word][1]))
            FAM.append(int(BristolNorms_1000311281[word][2]))
            num += 1
    if num != 0:
        feats[17] = np.mean(AoA)
        feats[18] = np.mean(IMG)
        feats[19] = np.mean(FAM)
        feats[20] = np.std(AoA)
        feats[21] = np.std(IMG)
        feats[22] = np.std(FAM)

    # 24.Average of V.Mean.Sum from Warringer norms
    # 25.Average of A.Mean.Sum from Warringer norms
    # 26.Average of D.Mean.Sum from Warringer norms
    # 27.Standard deviation of V.Mean.Sum from Warringer norms
    # 28.Standard deviation of A.Mean.Sum from Warringer norms
    # 29.Standard deviation of D.Mean.Sum from Warringer norms
    v_mean_sum = []
    a_mean_sum = []
    d_mean_sum = []
    num = 0
    for word in raw_comment_list:
        if word in Warringer_1000311281.keys():
            v_mean_sum.append(float(Warringer_1000311281[word][0]))
            a_mean_sum.append(float(Warringer_1000311281[word][1]))
            d_mean_sum.append(float(Warringer_1000311281[word][2]))
            num += 1
    if num != 0:
        feats[23] = np.mean(v_mean_sum)
        feats[24] = np.mean(a_mean_sum)
        feats[25] = np.mean(d_mean_sum)
        feats[26] = np.std(v_mean_sum)
        feats[27] = np.std(a_mean_sum)
        feats[28] = np.std(d_mean_sum)
    #print(feats)
    return feats


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    # read_csv()
    center_IDs = {}
    alt_IDs = {}
    left_IDs = {}
    right_IDs = {}
    f = open('/u/cs401/A1/feats/Center_IDs.txt')
    lines = f.readlines()
    for i in range(len(lines)):
        center_IDs[lines[i].rstrip('\r\n')] = i
    f.close()

    f = open('/u/cs401/A1/feats/Alt_IDs.txt')
    lines = f.readlines()
    for i in range(len(lines)):
        alt_IDs[lines[i].rstrip('\r\n')] = i
    f.close()

    f = open('/u/cs401/A1/feats/Left_IDs.txt')
    lines = f.readlines()
    for i in range(len(lines)):
        left_IDs[lines[i].rstrip('\r\n')] = i
    f.close()

    f = open('/u/cs401/A1/feats/Right_IDs.txt')
    lines = f.readlines()
    for i in range(len(lines)):
        right_IDs[lines[i].rstrip('\r\n')] = i
    f.close()

    center_feats = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')
    alt_feats = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')
    left_feats = np.load('/u/cs401/A1/feats/Left_feats.dat.npy')
    right_feats = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')

    data_len = len(data)
    for i in range(data_len):
    #for i in range(0, 5):
        comment = data[i]['body']
        feats[i][0:173] = extract1(comment)
        ID = data[i]['id']

        if data[i]['cat'] == 'Left':
            index = left_IDs[ID]
            feats[i][29:-1] = left_feats[index]
            feats[i][-1] = 0
        elif data[i]['cat'] == 'Center':
            index = center_IDs[ID]
            feats[i][29:-1] = center_feats[index]
            feats[i][-1] = 1
        elif data[i]['cat'] == 'Right':
            index = right_IDs[ID]
            feats[i][29:-1] = right_feats[index]
            feats[i][-1] = 2
        elif data[i]['cat'] == 'Alt':
            index = alt_IDs[ID]
            feats[i][29:-1] = alt_feats[index]
            feats[i][-1] = 3
        else:
            print("error")

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)

