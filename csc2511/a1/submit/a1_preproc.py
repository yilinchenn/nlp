import sys
import argparse
import os
import json
import html
import re
import spacy

indir_1000311281 = '/u/cs401/A1/data/';
stopwords_1000311281 = open('/u/cs401/Wordlists/StopWords')
stopwords_list_1000311281 = stopwords_1000311281.read().split('\n')

abbrev_1000311281 = open('/u/cs401/Wordlists/abbrev.english')
abbrev_str_1000311281 = abbrev_1000311281.read()
abbrev_list_1000311281 = abbrev_str_1000311281.split('\n')
temp_100311281 = abbrev_str_1000311281.replace('.', '').split('\n')  # no .
abbrev_wordlist_1000311281 = ['(?<!\s{0})'.format(x) for x in temp_100311281 if x is not '']
regex_str_1000311281 = re.sub(r"[|]", '', '|'.join(abbrev_wordlist_1000311281))
pattern_1000311281 = re.compile(regex_str_1000311281+"(\/\S+\s\.\/\S+\s)(?=[A-Z])") #(?<!\sst)(\/\S+\s\.\/\S+\s)(?=[A-Z])

nlp = spacy.load('en', disable=['parser', 'ner'])


def add_space(match):
    return match.group(1) + ' ' + match.group(2)

def add_newline(match):
    return match.group(0) + "\n"

def convert_lower(match):
    return match.group(0).lower()


def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    # print(comment)

    modComm = ''
    if 1 in steps:
        # remove newline character
        modComm = comment.replace("\n", " ")
    if 2 in steps:
        # replace HTML character code with ASCII
        # HTML to unicode to ascii
        modComm_Barray = html.unescape(modComm).encode('ascii', 'ignore')  # byte array
        modComm = modComm_Barray.decode('ascii')  # ascii string
    if 3 in steps:
        # remove html, eg., https://sdf, http://sdf, www.sdf
        # old = modComm
        # url = re.findall(r'(http:\/\/\S+|https:\/\/\S+|www\.\S+)', modComm)
        modComm = re.sub(r'(http:\/\/\S+|https:\/\/\S+|www\.\S+)', ' ', modComm)

    if 4 in steps:
        # split punctuation
        # modComm = 'asd? asdg... asdfg%&*adfg !!!!adfggf#$%^$%dfhdfh$%^& DR. e.g. i.e.'
        words = modComm.split(' ')
        result = ''
        for word in words:
            if word in abbrev_list_1000311281 or word == 'e.g.' or word == 'i.e.' or word == 'e.g...' or word == 'e.g.,':
                result += word + ' '
            else:
                padded_word = ''
                matches = re.findall(r"[!\"#$%&\(\)*+,-./:;<=>?@[\]^_`{|}~\\]+|[\w']+", word)
                for match in matches:
                    padded_word += (match + ' ')
                result += padded_word
        # print(result)
        modComm = result
    if 5 in steps:
        # split clitics
        # modComm = "adm's we've dogs' don't can't wouldn't y'all we'll we're, t'cha"
        modComm = re.sub(r"([\w]+)(\'s|\'S|\'ve|\'VE|\'m|\'M|\'ll|\'LL|\'re|\'RE|\'all|\'ALL|\'d|\'D|\'n|\'N)",
                         add_space, modComm)
        modComm = re.sub(r"(t\'|T\')([\w]+)", add_space, modComm)
        modComm = re.sub(r"(y|Y)(\'[\w]+)", add_space, modComm)
        modComm = re.sub(r"(\w[s|S])(\'\s) ", add_space, modComm)
        modComm = re.sub(r"([\w])(n\'t|N\'T)", add_space, modComm)
        # print(modComm)
    if 6 in steps:
        # add tag
        doc = nlp(modComm)
        comment = ''
        for token in doc:
            comment += (token.text + "/" + token.tag_ + ' ')
        modComm = comment
        # print(modComm)
    if 7 in steps:
        # reomve stopwords
        tokens = modComm.split(' ')
        result = ''
        for token in tokens:
            word = token.split('/')
            if word[0] not in stopwords_list_1000311281:
                result += (token + ' ')
        modComm = result
        # print(modComm)
    if 8 in steps:
        # lemmatization, keep the tags in step 6
        tokens = modComm.split(' ')
        tokens_str = ''
        tags = []
        for token in tokens:
            if token != '':
                word = token.split('/')
                tags.append(word[1])
                tokens_str += (word[0] + ' ')

        doc = nlp(tokens_str)
        # print(doc)
        comment = ''
        for i in range(len(tags)):
            if doc[i].lemma_[0] == '-' and doc[i].text[0] != '-':
                comment += (doc[i].text + "/" + tags[i] + ' ')
            elif doc[i].text.lower() == doc[i].lemma_:
                comment += (doc[i].text + "/" + tags[i] + ' ')
            else:
                comment += (doc[i].lemma_ + "/" + tags[i] + ' ')
        modComm = comment
        # print(modComm)
    if 9 in steps:
        # add newline between sentence
        modComm = re.sub(r"\s(\?|\!)\/\S+\s(?=[A-Z])", add_newline, modComm)
        modComm = re.sub(pattern_1000311281, add_newline, modComm)
        #print(modComm)
    if 10 in steps:
        # convert text to lower case
        modComm = re.sub(r"[\S]+(?=\/)", convert_lower, modComm)
        # print(modComm)


    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir_1000311281):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            data_len = len(data)
            for i in range(args.max):
                line_str = data[i + (args.ID[0] % data_len)]
                raw_line_json = json.loads(line_str)  # dict
                extracted_line = {}
                extracted_line['id'] = raw_line_json['id']
                extracted_line['body'] = raw_line_json['body']
                extracted_line['cat'] = file
                # print(extracted_line)

                extracted_line['body'] = preproc1(extracted_line['body'])
                allOutput.append(extracted_line)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
