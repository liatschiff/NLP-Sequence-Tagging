import math
global lambda1, lambda2, lambda3
import sys
from MLETrain import *
lambda1 = 0.8
lambda2 = 0.1
lambda3 = 0.1


def read_q_mle(path):
    q_mle_dict = {}
    with open(path ,'r') as f:
        lines = f.readlines()
        for line in lines:
            tags, count = line.split('\t')
            # tags = tags.split()
            q_mle_dict[tags] = int(count)
    return q_mle_dict


def read_e_mle(path):
    e_mle_dict = {}  # Contains all the options of (word, tag) and its count in the e_mle file.
    word_possible_tags = {}  # Contains the seen tags with each word or signature in the e.mle file.
    nb_of_tokens = 0  # The total number of seen words at the e.mle file without including the rare words (signatures).
    with open(path ,'r') as f:
        lines = f.readlines()
        for line in lines:
            word_tag , count = line.split('\t')
            word, tag = word_tag.split()
            e_mle_dict[(word,tag)] = int(count)

            if word[0] != '^':
                nb_of_tokens += int(count)

            if word not in word_possible_tags.keys():
                word_possible_tags[word] = set()
            
            word_possible_tags[word].add(tag)
            
    return e_mle_dict ,word_possible_tags, nb_of_tokens


def vocab_set(word_possible_tags):
    return set([word for word in word_possible_tags.keys() if word[0] != '^'])


def tokenizer(sentence):
    tokens = sentence.split()
    return tokens


def read_input_file(input_file):
    with open(input_file,'r') as f:
        sentences = f.readlines()
    return sentences


class Viterbi:
    def __init__(self, sentence, vocab, e_mle_dict, q_mle_dict, word_possible_tags, nb_of_tokens):
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
        
        # initialisation
        ###########################################
        word = self.sentence[0] if self.sentence[0] in self.vocab else find_signature(self.sentence[0])
        possible_tags = word_possible_tags[word] 
        ##########################################
        for i, tag in enumerate(possible_tags):
            e = math.log(self.getE(word, tag))
            q = math.log(self.getQ('START', 'START', tag))
            self.V[0][('START', tag)] = q + e
            self.B[0][('START', tag)] = 'START'

        self.recursion()
        self.backtracking()
        self.prediction_output = self.prediction_out()

    def list_tags(self):
        tags = set(list(zip(*self.e_mle_dict.keys()))[1])
        return list(tags)

    def getE(self,w1, t1):
        return self.e_mle_dict[(w1, t1)]/self.q_mle_dict[t1]

    def getQ(self,t1, t2, t3):
        '''
        return : lamda1 * prob(t3 |t1,t2) + lambda2 * prob(t3 | t2) + lambda3 * prob(t3)
        if sequence doesn't appears we define prob() = 1/number of sequences
        '''
        
        q_trigram = 0
        q_bigram = 0
        q_uni = 0

        if t1+" "+t2+" "+t3 in self.q_mle_dict.keys():
            q_trigram = self.q_mle_dict[t1+" "+t2+" "+t3] / self.q_mle_dict[t1+" "+t2]
        if t2+" "+t3 in self.q_mle_dict.keys():
            q_bigram = self.q_mle_dict[t2+" "+t3] / self.q_mle_dict[t2]
        if t3 in self.q_mle_dict.keys():
            q_uni = self.q_mle_dict[t3] / self.nb_of_tokens
        q = lambda1*q_trigram + lambda2*q_bigram + lambda3*q_uni
        return q

    def find_V_B(self, index, u, v):
        best_t = self.tags[0]
        best_value = - math.inf
        if index == (len(self.sentence)): # if tag_1 == 'END'
            e = 1
        else:
            word = self.sentence[index] if self.sentence[index] in self.vocab else find_signature(self.sentence[index])
            e = self.getE(word, v)

        e = math.log(e)

        if index < 2:
            tag = 'START'
            q = math.log(self.getQ(tag, u, v))
            new_val = self.V[index-1][(tag, u)] + q + e
            return new_val, tag
        
        possible_tags = set(list(zip(*self.V[index-1].keys()))[0])
        for tag in possible_tags:
            q = math.log(self.getQ(tag, u, v))
            new_val = self.V[index-1][(tag, u)] + q + e
            if new_val > best_value:
                best_value = new_val
                best_t = tag

        return best_value, best_t

    def recursion(self):
        ##################################
        for i in range(1, len(self.sentence)):
            word = self.sentence[i] if self.sentence[i] in self.vocab else find_signature(self.sentence[i])
            possible_tags_v = self.word_possible_tags[word]
            for v in possible_tags_v:
                possible_tags_u = set(list(zip(*self.V[i-1].keys()))[1])
                for u in possible_tags_u:
                    self.V[i][(u, v)], self.B[i][(u, v)] = self.find_V_B(i, u, v)

        possible_tags = set(list(zip(*self.V[len(self.sentence)-1].keys()))[1])
        for tag in possible_tags:
            self.V[len(self.sentence)][(tag,'END')], self.B[len(self.sentence)][(tag, 'END')] = self.find_V_B(len(self.sentence), tag, "END")

    def backtracking(self):
        tag_0, tag_1 = max(self.V[-1], key = self.V[-1].get)
        self.prediction_tag.insert(0, tag_0)
        for i in range(len(self.sentence),0,-1):
            new_tag_0 = self.B[i][(tag_0,tag_1)]
            self.prediction_tag.insert(0, new_tag_0)
            tag_1 = tag_0
            tag_0 = new_tag_0
    
    def prediction_out(self):
        word_tag = []
        for i, word in enumerate(self.sentence):
            word_tag.append(word + '/' + self.prediction_tag[i+1])
        return ' '.join(word_tag)


if __name__ == '__main__':
    input_file, q_mle, e_mle, output_file , extra_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    # input_file = 'data/ass1-tagger-dev-input'
    # output_file = 'viterbi_hmm_output.txt'
    # q_mle="q.mle.txt"
    # e_mle="e.mle.txt"
    q_mle_dict = read_q_mle(q_mle)
    e_mle_dict, word_possible_tags, nb_of_tokens = read_e_mle(e_mle)

    vocab = vocab_set(word_possible_tags)
    sentences = read_input_file(input_file)

    f_out = open(output_file, 'w')
    for sentence in sentences:
        tokens_sent = tokenizer(sentence)
        v_sent = Viterbi(tokens_sent, vocab, e_mle_dict, q_mle_dict, word_possible_tags, nb_of_tokens)
        # write to output file
        f_out.write(v_sent.prediction_output)
        f_out.write('\n')
    f_out.close()

