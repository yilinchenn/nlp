from math import exp
import re


def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on

    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.

    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.


	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""

    # TODO: Implement by student.

    candidate_words = re.findall(r"[\S]+", candidate)
    reference_wordlist = []  # [[SENTSTART, je, suis, faim, SENTEND], [SENTSTART, nous, sommes, faime, SENTEND]]
    lengthDiff = []
    sentenceLength = []
    for reference in references:
        word_list = re.findall(r"[\S]+", reference)
        reference_wordlist.append(word_list)
        sentenceLength.append(len(word_list))
        lengthDiff.append(abs(len(word_list) - len(candidate_words)))

    match = 0

    if n == 1:
        for word in candidate_words:
            for word_list in reference_wordlist:
                if word in word_list:
                    match += 1
                    break
        bleu_score = match / len(candidate_words)
        #print(match)
        #print(candidate_words)
    elif n == 2:
        for i in range(len(candidate_words)-1):
            old_match = match
            for word_list in reference_wordlist:
                if match != old_match:
                    break
                for j in range(len(word_list)-1):
                    if candidate_words[i] == word_list[j] and candidate_words[i+1] == word_list[j+1]:
                        match += 1
                        break

        bleu_score = match / (len(candidate_words)-1)
        # print(match)
        # print(candidate_words)
    elif n == 3:
        for i in range(len(candidate_words)-2):
            old_match = match
            for word_list in reference_wordlist:
                if match != old_match:
                    break
                for j in range(len(word_list)-2):
                    if candidate_words[i] == word_list[j] and candidate_words[i+1] == word_list[j+1] and candidate_words[i+2] == word_list[j+2]:
                        match += 1
                        break
        bleu_score = match / (len(candidate_words)-2)

    if brevity:
        index = lengthDiff.index(min(lengthDiff))
        brevity = sentenceLength[index] / len(candidate_words)
        if brevity < 1:
            BP = 1
        else:
            BP = exp(1 - brevity)

        bleu_score = bleu_score * BP

    return bleu_score

# if __name__ == "__main__":
#
#
#
#     candidate = "I am fear David"
#     references = ["I have fear David", "I am scared Dave", "I am afraid Dave"]
#     print(BLEU_score(candidate, references, 1, brevity=True))
#     print(BLEU_score(candidate, references, 2, brevity=False))
#     print(BLEU_score(candidate, references, 3, brevity=False))