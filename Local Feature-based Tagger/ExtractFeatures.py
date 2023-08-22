import time
from collections import Counter
import sys

def extract_data(sent, i, ti_1b, ti_2b, is_rare, label=""):
    i += 2
    sent = [*["<START>", "<START>"], *sent, *["<END>", "<END>"]] # concatenation: [<START>, <START>, sent, <end>, <end>]
    word = sent[i]
    form = "<UNK>" if is_rare else word # When the word is rare or doesn't appear in training
    wi_1a, wi_2a = sent[i+1], sent[i+2]
    wi_1b, wi_2b = sent[i-1], sent[i-2]
    suffix_1, prefix_1 = (word[-1:], word[:1])
    word_len = len(word)
    suffix_2, prefix_2 = (word[-2:], word[:2]) if word_len > 1 else ("", "")
    suffix_3, prefix_3 = (word[-3:], word[:3]) if word_len > 2 else ("", "")
    suffix_4, prefix_4 = (word[-4:], word[:4]) if word_len > 3 else ("", "")
    suffix_5, prefix_5 = (word[-5:], word[:5]) if word_len > 4 else ("", "")

    if label == "":  # Test time
        return {"form":form, "ti_1b":ti_1b, "ti_2b":ti_2b, "wi_1b":wi_1b, "wi_2b":wi_2b, "wi_1a":wi_1a, "wi_2a":wi_2a,
                "prefix_1":prefix_1, "prefix_2":prefix_2, "prefix_3":prefix_3, "prefix_4":prefix_4, "prefix_5":prefix_5,
                "suffix_1":suffix_1, "suffix_2":suffix_2, "suffix_3":suffix_3, "suffix_4":suffix_4, "suffix_5":suffix_5,
                }
    else:
        return {"label":label,"form":form, "ti_1b":ti_1b, "ti_2b":ti_2b, "wi_1b":wi_1b, "wi_2b":wi_2b, "wi_1a":wi_1a, "wi_2a":wi_2a,
                "prefix_1":prefix_1, "prefix_2":prefix_2, "prefix_3":prefix_3, "prefix_4":prefix_4, "prefix_5":prefix_5,
                "suffix_1":suffix_1, "suffix_2":suffix_2, "suffix_3":suffix_3, "suffix_4":suffix_4, "suffix_5":suffix_5,
                }


def main(corpus_file="data/ass1-tagger-train.txt", features_file="features_file.txt"):
    with open(corpus_file, mode='r') as t:
        sentences = []
        labels = []
        for line in t.readlines():
            sent = []
            tags_of_sent = []
            for w_t in line.split():
                last_slash = w_t.rindex('/')
                sent.append(w_t[:last_slash])
                tags_of_sent.append(w_t[last_slash+1:])
            sentences.append(sent)
            labels.append(tags_of_sent)
    train_words = Counter([word for sent in sentences for word in sent])
    rare_words = set([key for key, val in train_words.items() if val<2])

    features_corpus = []
    for s, sent in enumerate(sentences):
        features_sent = [extract_data(sent, 0, "<STARTAG>", "<STARTAG>", sent[0] in rare_words, labels[s][0])]
        for i in range(1, len(sent)):
            ti_1b, ti_2b = labels[s][i-1], features_sent[i - 1]["ti_1b"]
            features_sent.append(extract_data(sent, i, ti_1b, ti_2b, sent[i] in rare_words, labels[s][i]))
        features_corpus.append(features_sent)

    with open(features_file, mode='w') as f:
        for sent in features_corpus:
            for dict_word in sent:
                for key, val in dict_word.items():
                    f.write(f"{key}={val} ")
                f.write(f"\n")


if __name__ == '__main__':
    corpus_file, features_file = sys.argv[1], sys.argv[2]
    main(corpus_file, features_file)
