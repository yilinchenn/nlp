from preprocess import *
from lm_train import *
from math import log2
from collections import Counter

def goodTuring(sentence, LM):
    """
        Compute the LOG probability of a sentence with good-turing adjustment, given a language model

        INPUTS:
        sentence :      (string) The PROCESSED sentence whose probability we wish to compute
        LM :            (dictionary) The LM structure (not the filename)

        OUTPUT:
        log_prob :      (float) log probability of sentence
        """
    words = re.findall(r"[\S]+", sentence)
    log_prob = 0

    # count total number of bigrams
    bigram_count = 0
    for key1 in LM['bi']:
        bigram_count += sum(LM['bi'][key1].values())

    unigram_count = sum(LM['uni'].values())

    for t in range(len(words)):
        if t + 1 in range(len(words)):
            count_wt_wt1 = 0
            count_wt = 0

            if words[t] in LM['uni']:
                count_wt = LM['uni'][words[t]]

            N2 = count_wt + 1
            N1 = count_wt
            c = Counter(LM['uni'].values())
            count_N1 = c[N1]
            if N2 in c:
                count_N2 = c[N2]
            else:
                count_N2 = c[N1]

            if count_wt == 0:
                P_wt = N2 / unigram_count
            else:
                P_wt = N2 * count_N2 / (unigram_count * count_N1)

            if words[t] in LM['bi'] and words[t + 1] in LM['bi'][words[t]]:
                count_wt_wt1 = LM['bi'][words[t]][words[t + 1]]

            N2 = count_wt_wt1 + 1
            N1 = count_wt_wt1
            count_N1 = 0
            count_N2 = 0
            for word1 in LM['bi']:
                c = Counter(LM['bi'][word1].values())
                if N1 in c:
                    count_N1 += c[N1]
                if N2 in c:
                    count_N2 += c[N2]

            if count_wt_wt1 == 0:
                P_wt_wt1 = (N2 / bigram_count)
            else:
                # P_GT = (c_new * N2/N1) / N
                P_wt_wt1 = N2 * count_N2 / (bigram_count * count_N1)

            if P_wt_wt1 == 0 or P_wt == 0:
                # log(0) = -inf
                log_prob += float('-inf')
            else:
                log_prob += log2(P_wt_wt1 / P_wt)

    return log_prob

