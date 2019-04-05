from preprocess import *
from lm_train import *
from math import log2


def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
        Compute the LOG probability of a sentence, given a language model and whether or not to
        apply add-delta smoothing

        INPUTS:
        sentence :      (string) The PROCESSED sentence whose probability we wish to compute
        LM :            (dictionary) The LM structure (not the filename)
        smoothing : (boolean) True for add-delta smoothing, False for no smoothing
        delta :         (float) smoothing parameter where 0<delta<=1
        vocabSize :     (int) the number of words in the vocabulary

        OUTPUT:
        log_prob :      (float) log probability of sentence
        """

    # TODO: Implement by student.
    words = re.findall(r"[\S]+", sentence)
    log_prob = 0
    # shall we count the prob if it is not a word?????????????
    if smoothing:
        for t in range(len(words)-1):
            count_wt = 0
            count_wt_wt1 = 0
            if words[t] in LM['uni']:
                count_wt = LM['uni'][words[t]]
            if words[t] in LM['bi'] and words[t+1] in LM['bi'][words[t]]:
                count_wt_wt1 = LM['bi'][words[t]][words[t+1]]

            current_prob = log2((count_wt_wt1 + delta)/(count_wt + delta * vocabSize))
            log_prob += current_prob
    else:
        for t in range(len(words)):
            if t+1 in range(len(words)):
                count_wt = 0
                count_wt_wt1 = 0
                if words[t].strip() == '':
                    continue

                if words[t] in LM['uni']:
                    count_wt = LM['uni'][words[t]]

                if words[t] in LM['bi'] and words[t+1] in LM['bi'][words[t]]:
                    count_wt_wt1 = LM['bi'][words[t]][words[t+1]]

                if count_wt == 0 or count_wt_wt1 == 0:
                    # log(0) = -inf
                    log_prob += float('-inf')
                else:
                    log_prob += log2(count_wt_wt1/count_wt)

    return log_prob

# if __name__ == "__main__":
#     lm = lm_train("/u/cs401/A2_SMT/data/Toy/", "e", "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/lm")
#
#     f = open("/u/cs401/A2_SMT/data/Toy/toy.e")
#
#     print(len(lm['uni']))
#     print(lm)
#
#     sentences = f.readlines()
#     for sentence in sentences:
#         proc_sentence = preprocess(sentence, "e")
#         prob = log_prob(proc_sentence, lm, True, delta = 0.0001, vocabSize =len(lm['uni']))
#         print(proc_sentence)
#         print(prob)
#         break