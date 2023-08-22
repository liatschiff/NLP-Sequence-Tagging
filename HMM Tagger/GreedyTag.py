import math
import sys
from MLETrain import *
global lambda1, lambda2, lambda3
lambda1 = 0.6
lambda2 = 0.2
lambda3 = 0.2



def read_q_mle(path):
    q_mle_dict = {}
    with open(path ,'r') as f:
        lines = f.readlines()
        for line in lines:
            tags, count = line.split('\t')
            # tags = tuple(tags.split())
            q_mle_dict[tags] = int(count)
    return q_mle_dict


def read_e_mle(path):
    e_mle_dict = {}  # Contains all the options of (word, tag) and its count in the e_mle file.
    word_possible_tags = {}  # Contains the seen tags with each word or signature in the e.mle file.
    nb_of_tokens = 0  # The total number of seen words at the e.mle file without including the rare words (signatures).
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            word_tag, count = line.split('\t')
            word, tag = word_tag.split()
            e_mle_dict[(word, tag)] = int(count)

            if word[0] != '^':
                nb_of_tokens += int(count)

            if word not in word_possible_tags.keys():
                word_possible_tags[word] = set()
            
            word_possible_tags[word].add(tag)
            
    return e_mle_dict, word_possible_tags, nb_of_tokens


def vocab_set(word_possible_tags):
    return set([word for word in word_possible_tags.keys() if word[0] != '^'])


def tokenizer(sentence):
    return sentence.split()


def read_input_file(input_file):
    with open(input_file,'r') as f:
        sentences = f.readlines()
    return sentences


class Greedy:
    def __init__(self, sentence,  vocab, e_mle_dict,q_mle_dict, word_possible_tags,nb_of_tokens):
        '''
        input: sentence is list of tokens of one sentence
        '''
        self.vocab = vocab
        self.e_mle_dict = e_mle_dict
        self.q_mle_dict = q_mle_dict
        self.word_possible_tags = word_possible_tags
        self.nb_of_tokens = nb_of_tokens
        self.sentence = sentence
        self.tags = self.list_tags()
        self.prediction_tag = []
        # for each word we create dictionary d , each key will represent tag and value probability of the tag.
        self.V = [dict() for i in range(len(sentence)+1)]
        # for each word we create dictionary d, each key will represent tag and value will be the best tag of the last word.
        self.B = [dict() for i in range(len(sentence)+1)]

        self.greedy_prediction()
        self.prediction_output = self.prediction_out()

    def greedy_prediction(self):
        self.prediction_tag = []
        prev_prev_tag = 'START'
        prev_tag = 'START'
        for i,tokens in enumerate(self.sentence):
            word = tokens if tokens in self.vocab else find_signature(tokens)
            if word not in word_possible_tags.keys():
                word = '^UNK'
            possible_tags = word_possible_tags[word] 
            p_ti = -math.inf
            tag_i = None
            for tag in possible_tags:
                # if i == 0:
                #     e = self.getE(tokens.lower(), tag) + self.getE(tokens, tag)
                # else:
                e = math.log(self.getE(word, tag))
                g_t = e + math.log(self.getQ(prev_prev_tag, prev_tag, tag))
                if g_t > p_ti:
                    p_ti = g_t
                    tag_i = tag
            
            self.prediction_tag.append(tag_i)
            prev_prev_tag = prev_tag
            prev_tag = tag_i

    def prediction_out(self):
        word_tag = []
        for i, word in enumerate(self.sentence):
            word_tag.append(word + '/' + self.prediction_tag[i])
        return ' '.join(word_tag)

    def list_tags(self):
        tags = set(list(zip(*self.e_mle_dict.keys()))[1])
        return list(tags)

    def getE(self,w1, t1):
        return self.e_mle_dict[(w1,t1)]/self.q_mle_dict[t1]

    def getQ(self,t1, t2, t3):
        '''
        return : lamda1 * prob(t3 |t1,t2) + lambda2 * prob(t3 | t2) + lambda3 * prob(t3)
        if sequence doesn't appears we define prob() = 1/number of sequences
        '''
        
        q_trigram = 0
        q_bigram = 0
        q_uni = 0

        if t1+" "+t2+" "+t3 in self.q_mle_dict.keys():
            q_trigram = self.q_mle_dict[t1+" "+t2+" "+t3]/self.q_mle_dict[t1+" "+t2]
        if t2+" "+t3 in self.q_mle_dict.keys():
            q_bigram = self.q_mle_dict[t2+" "+t3]/self.q_mle_dict[t2]
        if t3 in self.q_mle_dict.keys():
            q_uni = self.q_mle_dict[t3] / self.nb_of_tokens
        q = lambda1*q_trigram + lambda2*q_bigram + lambda3*q_uni

        return q


if __name__ == '__main__':
    input_file, q_mle, e_mle, output, extra_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    q_mle_dict = read_q_mle(q_mle)
    e_mle_dict, word_possible_tags, nb_of_tokens = read_e_mle(e_mle)

    vocab = vocab_set(word_possible_tags)
    sentences = read_input_file(input_file)

    f_out = open(output, 'w')
    for i, sentence in enumerate(sentences):
        tokens_sent = tokenizer(sentence)
        g_sent = Greedy(tokens_sent, vocab, e_mle_dict, q_mle_dict, word_possible_tags, nb_of_tokens)
        # write to output file
        f_out.write(g_sent.prediction_output)
        f_out.write('\n')
    f_out.close()

