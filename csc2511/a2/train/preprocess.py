import re

def add_padding(match):
    return match.group(1) + ' '
def add_space(match):
    return match.group(1) + ' ' + match.group(2)
def convert_lower(match):
    return match.group(0).lower()

def preprocess(in_sentence, language):
    """
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation

        INPUTS:
        in_sentence : (string) the original sentence to be processed
        language        : (string) either 'e' (English) or 'f' (French)
                                   Language of in_sentence

        OUTPUT:
        out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function
    # separate sentence-final punctuation (sentences have already been determined for you),
    # commas, colons and semicolons, parentheses, dashes between parentheses, mathematical
    # operators (e.g., +, -, <, >,=), and quotation marks
    in_sentence = in_sentence.strip()
    in_sentence = re.sub(r"[\S]+", convert_lower, in_sentence)

    tokens = re.findall(r"[.!?,:;\(\)\"-+<>=]|[\w']+", in_sentence)
    sentence_proc = ""
    for token in tokens:
        sentence_proc += (token + ' ')

    if language == 'f':
        #print(sentence_proc)
        sentence_proc = re.sub(r"(l\'|L\'|t\'|T\'|j\'|J\'|qu\'|Qu\')([\w]+)", add_space, sentence_proc)
        sentence_proc = re.sub(r"([\w]+)(\'on|\'il)", add_space, sentence_proc)


    out_sentence = "SENTSTART " + sentence_proc + "SENTEND"

    return out_sentence


if __name__ == "__main__":
    # o = preprocess("l'election je t'aime.", "f")
    # print(o)
    o = preprocess("sdfaqu't?", "f")
    print(o)
    # o = preprocess("ENGLISH   P.M.  3-5  I'M.)   ", "e")
    # print(o)
    e = open("/u/cs401/A2_SMT/data/Hansard/Training/hansard.36.1.house.debates.192.e")
    f = open("/u/cs401/A2_SMT/data/Hansard/Training/hansard.36.1.house.debates.192.f")

    # sentences = f.readlines()
    # for sentence in sentences:
    #     print(preprocess(sentence, "f"))

    # sentences = e.readlines()
    # for sentence in sentences:
    #     print(preprocess(sentence, 'e'))

    # print(len(sentences))