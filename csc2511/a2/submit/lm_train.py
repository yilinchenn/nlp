from preprocess import *
import pickle
import os


def lm_train(data_dir, language, fn_LM):
    """
        This function reads data from data_dir, computes unigram and bigram counts,
        and writes the result to fn_LM

        INPUTS:

    data_dir    : (string) The top-level directory continaing the data from which
                                        to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
        language        : (string) either 'e' (English) or 'f' (French)
        fn_LM           : (string) the location to save the language model once trained

    OUTPUT

        LM                      : (dictionary) a specialized language model

        The file fn_LM must contain the data structured called "LM", which is a dictionary
        having two fields: 'uni' and 'bi', each of which holds sub-structures which
        incorporate unigram or bigram counts

        e.g., LM['uni']['word'] = 5             # The word 'word' appears 5 times
                  LM['bi']['word']['bird'] = 2  # The bigram 'word bird' appears 2 times.
    """

    language_model = {}
    language_model['uni'] = {}
    language_model['bi'] = {}

    # open file, read sentence and call preprocess
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if language == 'e' and file.split(".")[-1] == 'e':
                fullFile = os.path.join(subdir, file)
                f = open(fullFile)
                #print(fullFile)
                sentences = f.readlines()
                for sentence in sentences:
                    sentence = sentence.strip()
                    # skip empty line
                    if sentence == "":
                        continue
                    processed_sentence = preprocess(sentence, language)
                    words = re.findall(r"[\S]+", processed_sentence)
                    for i in range(len(words)):
                        if words[i] in language_model['uni']:
                            language_model['uni'][words[i]] += 1
                        else:
                            language_model['uni'][words[i]] = 1

                        if i + 1 < len(words):
                            if words[i] in language_model['bi'] and words[i + 1] in language_model['bi'][words[i]]:
                                language_model['bi'][words[i]][words[i + 1]] += 1
                            elif words[i] in language_model['bi'] and words[i + 1] not in language_model['bi'][
                                words[i]]:
                                language_model['bi'][words[i]][words[i + 1]] = 1
                            elif words[i] not in language_model['bi']:
                                language_model['bi'][words[i]] = {}
                                language_model['bi'][words[i]][words[i + 1]] = 1

            elif language == 'f' and file.split(".")[-1] == 'f':
                fullFile = os.path.join(subdir, file)
                f = open(fullFile)
                #print(fullFile)
                sentences = f.readlines()
                for sentence in sentences:
                    sentence = sentence.strip()
                    # skip empty line
                    if sentence == "":
                        continue
                    processed_sentence = preprocess(sentence, language)
                    words = re.findall(r"[\S]+", processed_sentence)
                    for i in range(len(words)):
                        if words[i] in language_model['uni']:
                            language_model['uni'][words[i]] += 1
                        else:
                            language_model['uni'][words[i]] = 1

                        if i + 1 < len(words):
                            if words[i] in language_model['bi'] and words[i + 1] in language_model['bi'][words[i]]:
                                language_model['bi'][words[i]][words[i + 1]] += 1
                            elif words[i] in language_model['bi'] and words[i + 1] not in language_model['bi'][
                                words[i]]:
                                language_model['bi'][words[i]][words[i + 1]] = 1
                            elif words[i] not in language_model['bi']:
                                language_model['bi'][words[i]] = {}
                                language_model['bi'][words[i]][words[i + 1]] = 1


    # Save Model
    with open(fn_LM + '.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return language_model

# if __name__ == "__main__":
#     print(lm_train("/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/test/", "e", "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/LM_task2"))
#     # lm_train("/u/cs401/A2_SMT/data/Hansard/Training/", "f", "/h/u8/c8/00/chenyil2/Desktop/csc2511/a2/lm")
