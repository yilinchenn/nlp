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

    words = in_sentence.split(' ')
    sentence_proc = ""
    for word in words:
        token = re.sub(r"[\.\!\?\,\:\;\(\)\"\-\+\<\>\=]|[^\.\!\?\,\:\;\(\)\"\-\+\<\>\=]+|[\w]+[^\.\!\?\,\:\;\(\)\"\-\+\<\>\=]+", lambda mat: mat.group(0) + " ", word)
        sentence_proc += (token + ' ')

    # fix spaces
    words = re.findall(r"[\S]+", sentence_proc)
    sentence_proc = ""
    for word in words:
        sentence_proc += (word + ' ')

    #print(sentence_proc)
    if language == 'f':
        #print(sentence_proc)
        sentence_proc = re.sub(r"(l\'|t\'|j\'|c\'|qu\'|puisqu\'|lorsqu\')([\w]+)", add_space, sentence_proc)


    out_sentence = "SENTSTART " + sentence_proc + "SENTEND"

    return out_sentence


# if __name__ == "__main__":
#     # english_path = "/u/cs401/A2_SMT/data/Hansard/Training/hansard.36.1.house.debates.192.e"
#     # french_path = "/u/cs401/A2_SMT/data/Hansard/Training/hansard.36.1.house.debates.192.f"
#     # french_file = open(french_path, 'r')
#     # french_file = french_file.read().split('\n')
#     # english_file = open(english_path, 'r')
#     # english_file = english_file.read().split('\n')
#     #
#     # # for line in english_file:
#     # #     print(line)
#     # #     print(preprocess(line, 'e'))
#     # for line in french_file:
#     #     print(line)
#     #     print(preprocess(line, 'f'))
#     #
#     # #print(preprocess("(La seance est levee a 19 h 23.)", "f"))
#     print(preprocess("Mr.Dick Harris (PrinceGeorge-Bulkley Valley, Ref.):", "e"))
#     print(preprocess("\n\n", "e"))
#     # preprocess_me("I have never seen them so low.", "e")
#     # preprocess_me("This was followed by the question ``Where do you want us to go?'' ", "e")
#     # preprocess_me("Advertisement revenues represent 60 % of Canadian periodical revenues. ", "e")
#     # preprocess_me("That number has dropped (*to. 38%.", "e")
