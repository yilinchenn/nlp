from log_prob import *
from preprocess import *
from goodTuring_bonus import *
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
	Computes the preplexity of language model given a test corpus
	
	INPUT:
	
	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	"""

    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])

    print(vocab_size)
    
    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue
        
        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            # tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)
            tpp = goodTuring(processed_line, LM)
            
            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
        pp = 2**(-pp/N)
    return pp

#test
if __name__ == "__main__":
    # lm = lm_train("/u/cs401/A2_SMT/data/Hansard/Training/", "e", "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/lm_ma")
    #
    # # print(preplexity(lm, "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/test/", "e", False, delta = 0.0001))
    # # print(preplexity(lm, "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/test/", "e", True, delta=0.0001))
    #
    # print(preplexity(lm, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", False, delta = 0.0001))
    # print(preplexity(lm, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", True, delta=0.01))
    # print(preplexity(lm, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", True, delta=0.1))
    # print(preplexity(lm, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", True, delta=0.5))
    # print(preplexity(lm, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", True, delta=1))

    lm = lm_train("/u/cs401/A2_SMT/data/Hansard/Training/", "e", "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/lm_ma")

    # print(preplexity(lm, "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/test/", "e", False, delta = 0.0001))
    # print(preplexity(lm, "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/test/", "e", True, delta=0.0001))

    print(preplexity(lm, "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/test/", "e", False, delta = 0.0001))
