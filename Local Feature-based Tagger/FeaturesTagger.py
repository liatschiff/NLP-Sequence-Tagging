import time
import pickle
from ExtractFeatures import extract_data
import sys

def read_input_file(input_file_name):
    with open(input_file_name, mode='r') as t:
        corpus = []
        for line in t.readlines():
            corpus.append(line.split())
    return corpus, len(corpus)


def write_output_file(output_file_name, prediction_output_dict):
    with open(output_file_name, mode='w') as t:
        for idx_sent in range(len(prediction_output_dict)):
            prediction_idx_sent = prediction_output_dict[idx_sent]
            first_word, first_tag = prediction_idx_sent[0]
            t.write(str(first_word + "/" + first_tag))
            for word, tag in prediction_idx_sent[1:]:
                t.write(str(" " + word + "/" + tag))
            t.write(str("\n"))


def main(input_file_name="data/ass1-tagger-test-input.txt", model_file_name="model_file", feature_map_file="feature_map_file", output_file="feats-predictions.txt"):
    corpus, sentences_amount = read_input_file(input_file_name)

    with open(feature_map_file, 'rb') as map_f:
        DictVectorizer = pickle.load(map_f)
        seen_words = pickle.load(map_f)

    with open(model_file_name, 'rb') as mod_f:
        model = pickle.load(mod_f)

    keys = [idx_sent for idx_sent in range(sentences_amount)]
    values = [[("<STARTAG>", "<STARTAG>")] for i in range(sentences_amount)]
    prediction_dict = dict(zip(keys, values))
    prediction_output_dict = {k: [] for k in keys}

    for idx_word in range(max([len(s) for s in corpus])):
        x = []
        sentences = [(i, sent) for i,sent in enumerate(corpus) if len(sent)>idx_word]

        # only sentences that their length is at least idx_word
        for idx_sent, sent in sentences:
            # prev_information = tuple(ti_1b, ti_2b)
            prev_information = prediction_dict[idx_sent][-1]
            # features_sent.append(extract_data(sent, i, ti_1b, ti_2b, sent[i] in rare_words, labels[i]))
            word_features_dict = extract_data(sent, idx_word, prev_information[0], prev_information[1], (not sent[idx_word] in seen_words))
            x.append(word_features_dict)
        x = DictVectorizer.transform(x)
        tags_of_idx_words = model.predict(x)

        for i, idx_sent_sent in enumerate(sentences):
            idx_sent, sent = idx_sent_sent
            prediction_output_dict[idx_sent].append((sent[idx_word], tags_of_idx_words[i])) # for the ouput file {sent_i:[tuple(word,tag),...],...}
            prev_information = prediction_dict[idx_sent][-1]
            prediction_dict[idx_sent].append((tags_of_idx_words[i], prev_information[0]))

    write_output_file(output_file, prediction_output_dict)


if __name__ == '__main__':
    start = time.time()
    input_file_name, model_file_name, feature_map_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    main(input_file_name, model_file_name, feature_map_file, output_file)
    end = time.time()
    print("%.2f sec" % (end-start))