import os
import numpy as np
import fnmatch
import re

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    n = len(r)
    m = len(h)
    R = np.zeros((n + 1, m + 1))
    B = np.zeros((n + 1, m + 1))
    R[0] = np.arange(m + 1)
    R[:, 0] = np.arange(n + 1)
    # up is 0, left is 1, up-left is 2, up-left but match is 3
    B[0] = 1
    B[:, 0] = 0

    # print("%d, %d"%(n, m))
    # print(B)
    # print(R)

    for i in range(1, n+1):
        for j in range(1, m+1):
            deletion = R[i - 1, j] + 1

            if r[i - 1] == h[j - 1]:
                sub = R[i - 1, j - 1]
            else:
                sub = R[i - 1, j - 1] + 1

            ins = R[i, j - 1] + 1

            # if i == 3 and j == 1:
            #     print("%d, %d, %d"%(deletion, sub, ins))

            R[i, j] = min(deletion, sub, ins)
            if R[i, j] == deletion:
                B[i, j] = 0  # up
            elif R[i, j] == ins:
                B[i, j] = 1  # left
            elif R[i, j] == sub and R[i - 1, j - 1] != R[i, j]:
                B[i, j] = 2  # up-left
            else:
                B[i, j] = 3  # up-left but match (correct substitution)

    #print(B)
    # print(R)

    # backtrace
    backtrace = [B[n, m]]
    current_pos = (n, m)
    nS, nI, nD = 0, 0, 0
    #exit()

    while current_pos[0] != 0 or current_pos[1] != 0:
        # print(i)
        if backtrace[-1] == 3:
            current_pos = (current_pos[0] - 1, current_pos[1] - 1)
            backtrace.append(B[current_pos[0], current_pos[1]])
        elif backtrace[-1] == 2:
            current_pos = (current_pos[0] - 1, current_pos[1] - 1)
            backtrace.append(B[current_pos[0], current_pos[1]])
            nS += 1
        elif backtrace[-1] == 1:
            current_pos = (current_pos[0], current_pos[1] - 1)
            backtrace.append(B[current_pos[0], current_pos[1]])
            nI += 1
        else:
            current_pos = (current_pos[0] - 1, current_pos[1])
            backtrace.append(B[current_pos[0], current_pos[1]])
            nD += 1

    # print(backtrace)
    if n != 0:
        wer = (nS + nI + nD) / n
    else:
        wer = float('Inf')

    return (wer, nS, nI, nD)


def preProcess(lines):
    # each element is a list of words
    processed_lines = []
    for i in range(len(lines)):
        if lines[i] != '':
            line = ' '.join(lines[i].strip().split()[2:])
            line = line.lower()
            # line = re.sub(r"\<\S*?\>", '', line)
            line = re.sub(r"[!\"#$%&\(\)*+,-./:;=?@\^_`{|}~\\]+", '', line)

            # TODO: REMOVE []????????
            #line = re.sub(r"\[\S*?\]", '', line)
            wordlist = line.split()
            # print(lines[i])
            # print(line)
            # print("\n")
            processed_lines.append(wordlist)

    return processed_lines


if __name__ == "__main__":
    # print(Levenshtein("who is there".split(), "is there".split()))
    # print(Levenshtein("who is there".split(), "".split()))
    # print(Levenshtein("".split(), "who is there".split()))

    out = open("asrDiscussion.txt", "w")
    wer_google = []
    wer_kaldi = []
    print(dataDir)

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*txt')
            print(files)
            for file in files:
                if "Google" in file:
                    f = open(os.path.join(dataDir, speaker, file), "r")
                    lines = f.readlines()
                    processed_google = preProcess(lines)
                elif "Kaldi" in file:
                    f = open(os.path.join(dataDir, speaker, file), "r")
                    lines = f.readlines()
                    processed_kaldi = preProcess(lines)
                else:
                    f = open(os.path.join(dataDir, speaker, file), "r")
                    lines = f.readlines()
                    processed_ref = preProcess(lines)

            #print(len(processed_ref))

            for i in range(len(processed_ref)):
                kaldi_result = Levenshtein(processed_ref[i], processed_kaldi[i])
                out.write("%s Kaldi   %d %f S:%d, I:%d, D:%d\n" % (
                speaker, i, kaldi_result[0], kaldi_result[1], kaldi_result[2], kaldi_result[3]))
                wer_kaldi.append(kaldi_result[0])

                google_result = Levenshtein(processed_ref[i], processed_google[i])
                out.write("%s Google  %d %f S:%d, I:%d, D:%d\n" % (
                speaker, i, google_result[0], google_result[1], google_result[2], google_result[3]))
                wer_google.append(google_result[0])

            #break

    out.write("Google mean: %f, std: %f Kaldi mean: %f, std: %f\n" %(np.mean(wer_google), np.std(wer_google), np.mean(wer_kaldi), np.std(wer_kaldi)))
    out.close()