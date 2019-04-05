from lm_train import *
from log_prob import *
from preprocess import *
from math import log
from collections import Counter
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    
    # Read training data
    eng, fre = read_hansard(train_dir, num_sentences)

    # Initialize AM uniformly
    AM = initialize(eng, fre)
    
    # Iterate between E and M steps
    for i in range(max_iter):
        AM = em_step(AM, eng, fre)

    #print("*************************************")
    # print(AM)

    # Save Model
    with open(fn_AM + '.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""

    total_sentences = 0
    eng = []
    fre = []

    for subdir, dirs, files in os.walk(train_dir):
        if total_sentences == num_sentences:
            break
        for file in files:
            if total_sentences == num_sentences:
                break

            if file.split(".")[-1] == 'e':
                # print("total sentences")
                # print(total_sentences)
                # print(file)
                fullFile_eng = os.path.join(subdir, file)
                f_eng = open(fullFile_eng)
                fre_file = file[0:-1]+'f'
                # print(fre_file)
                fullFile_fre = os.path.join(subdir, fre_file)
                f_fre = open(fullFile_fre)

                eng_training = f_eng.readlines()
                fre_training = f_fre.readlines()
                # print(len(eng_training))

                for i in range(len(eng_training)):
                    if eng_training[i].strip() != "" and total_sentences < num_sentences:
                        eng_sen = preprocess(eng_training[i], "e")
                        # eng_sen = eng_training[i]
                        eng.append(re.findall(r"[\S]+", eng_sen))
                        fre_sen = preprocess(fre_training[i], "f")
                        # fre_sen = fre_training[i]
                        fre.append(re.findall(r"[\S]+", fre_sen))
                        total_sentences += 1

    # print(eng)
    # print(fre)
    return eng, fre

def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    AM = {}
    eng_list = {} # key is english word, value is a list of words in corresponding french sentences
    eng_count = {}

    for i in range(len(eng)):
        eng_wordlist = eng[i]
        for eng_word in eng_wordlist:
            if eng_word not in eng_list:
                eng_list[eng_word] = []
                eng_list[eng_word].extend(fre[i])
            else:
                eng_list[eng_word].extend(fre[i])

    for eng_word in eng_list:
        #print(eng_word)
        #print(eng_list[eng_word])
        eng_count[eng_word] = len(Counter(eng_list[eng_word])) - 2

    for i in range(len(eng)):
        eng_wordlist = eng[i]
        fre_wordlist = fre[i]
        for eng_word in eng_wordlist:
            if eng_word != 'SENTSTART' and eng_word != 'SENTEND':
                #print(eng_word)
                for fre_word in fre_wordlist:
                    if fre_word != "SENTSTART" and fre_word != "SENTEND":
                        if eng_word in AM and fre_word not in AM[eng_word]:
                            AM[eng_word][fre_word] = 1 / eng_count[eng_word]
                        elif eng_word not in AM:
                            AM[eng_word] = {}
                            AM[eng_word][fre_word] = 1 / eng_count[eng_word]

    AM["SENTSTART"] = {}
    AM["SENTEND"] = {}
    AM["SENTSTART"]["SENTSTART"] = 1
    AM["SENTEND"]["SENTEND"] = 1
    #print(AM)
    return AM
    
def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
    tcount = {}
    total = {}

    for key_eng in t.keys():
        if key_eng == "SENTSTART" or key_eng == "SENTEND":
            continue

        total[key_eng] = 0
        tcount[key_eng] = {}
        for key_fre in t[key_eng]:
            if key_fre == "SENTSTART" or key_fre == "SENTEND":
                continue

            tcount[key_eng][key_fre] = 0

    for i in range(len(eng)):
        # each unique word in sentence
        eng_wordlist = Counter(eng[i])
        fre_wordlist = Counter(fre[i])
        for fre_word in fre_wordlist:
            if fre_word == "SENTSTART" or fre_word == "SENTEND":
                continue

            denom_c = 0
            for eng_word in eng_wordlist:
                if eng_word == "SENTSTART" or eng_word == "SENTEND":
                    continue

                denom_c += t[eng_word][fre_word] * fre_wordlist[fre_word]

            for eng_word in eng_wordlist:
                if eng_word == "SENTSTART" or eng_word == "SENTEND":
                    continue

                tcount[eng_word][fre_word] += t[eng_word][fre_word] * fre_wordlist[fre_word] * eng_wordlist[eng_word] / denom_c
                total[eng_word] += t[eng_word][fre_word] * fre_wordlist[fre_word] * eng_wordlist[eng_word] / denom_c
    #print(total)

    for eng_word in total.keys():
        for fre_word in tcount[eng_word].keys():
            t[eng_word][fre_word] = tcount[eng_word][fre_word] / total[eng_word]

    return t

# if __name__ == "__main__":
#     AM = align_ibm1("/u/cs401/A2_SMT/data/Toy/", 4, 3, "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/am")
