#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

from decode import *
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    if use_cached:
        #print(fn_LM)
        with open(fn_LM + '.pickle', 'rb') as handle:
            lm = pickle.load(handle)
        return lm
    else:
        lm = lm_train(data_dir, language, fn_LM)
        return lm


def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    if use_cached:
        #print(fn_AM)
        with open(fn_AM + '.pickle', 'rb') as handle:
            AM = pickle.load(handle)
        return AM
    else:
        AM = align_ibm1(data_dir, num_sent, max_iter, fn_AM)
        return AM

def _get_brevity(candidate, references):
    candidate_words = re.findall(r"[\S]+", candidate)
    reference_wordlist = []  # [[SENTSTART, je, suis, faim, SENTEND], [SENTSTART, nous, sommes, faime, SENTEND]]
    lengthDiff = []
    sentenceLength = []
    for reference in references:
        word_list = re.findall(r"[\S]+", reference)
        reference_wordlist.append(word_list)
        sentenceLength.append(len(word_list))
        lengthDiff.append(abs(len(word_list) - len(candidate_words)))

    index = lengthDiff.index(min(lengthDiff))
    brevity = sentenceLength[index] / len(candidate_words)
    if brevity < 1:
        BP = 1
    else:
        BP = exp(1 - brevity)

    return BP

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    evals = []
    for i in range(len(eng_decoded)):
        candidate = eng_decoded[i]
        references = [eng[i], google_refs[i]]
        bleu_score = 0
        if n == 1:
            bleu_score = BLEU_score(candidate, references, 1, brevity=True)
        elif n == 2:
            p1 = BLEU_score(candidate, references, 1, brevity=False)
            p2 = BLEU_score(candidate, references, 2, brevity=False)
            brevity = _get_brevity(candidate, references)
            bleu_score = brevity * ((p1*p2)**(1/n))
        elif n == 3:
            p1 = BLEU_score(candidate, references, 1, brevity=False)
            p2 = BLEU_score(candidate, references, 2, brevity=False)
            p3 = BLEU_score(candidate, references, 3, brevity=False)
            brevity = _get_brevity(candidate, references)
            bleu_score = brevity * ((p1*p2*p3)**(1/n))
        evals.append(bleu_score)
    return evals
   

def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    

    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    LM = _getLM("/u/cs401/A2_SMT/data/Hansard/Training/", "e", "task2", use_cached=True)
    AMs = {}
    AM_1k = _getAM("/u/cs401/A2_SMT/data/Hansard/Training/", 1000, 100, "am", use_cached=False)
    AM_10k = _getAM("/u/cs401/A2_SMT/data/Hansard/Training/", 10000, 100, "am_10k", use_cached=True)
    AM_15k = _getAM("/u/cs401/A2_SMT/data/Hansard/Training/", 15000, 100, "am_15k", use_cached=True)
    AM_30k = _getAM("/u/cs401/A2_SMT/data/Hansard/Training/", 30000, 100, "am_30k", use_cached=True)
    AMs[1000] = AM_1k
    AMs[10000] = AM_10k
    AMs[15000] = AM_15k
    AMs[30000] = AM_30k

    f_candidate = open("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f")
    f_hansard = open("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e")
    f_google = open("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e")
    candidates = []
    hansard = []
    google = []
    for i in range(25):
        candidates.append(preprocess(f_candidate.readline(), 'f'))
        hansard.append(preprocess(f_hansard.readline(), 'e'))
        google.append(preprocess(f_google.readline(), 'e'))

    # print(len(hansard))
    # print(len(google))

    f = open("Task5.txt", 'w+')
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    for i in AMs:
        AM = AMs[i]
        f.write("\n### Evaluating AM model: number of sentenses = %d ### \n"%i)

        # Decode using AM #
        # Eval using 3 N-gram models #

        decoded_sen = []
        # print(candidates[0])
        # print(decode(candidates[0], LM, AM))
        for j in range(25):
            decoded_sen.append(decode(candidates[j], LM, AM))

        # print(len(decoded_sen))
        # for j in range(25):
        #     print(decoded_sen[j])
        #     print(hansard[j])
        #     print(google[j])
        #
        # print("*******************************************************************")

        all_evals = []
        for n in range(1, 4):
            f.write("\nBLEU scores with N-gram (n) = %d: "%n)

            evals = _get_BLEU_scores(decoded_sen, hansard, google, n)
            for v in evals:
                f.write("\t{%1.4f}"%v)
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)