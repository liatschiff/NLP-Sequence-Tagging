import sys
import re
RARE_NUMBER = 1


# util
def writing_to_file(words_tags, d_signature, file_to_write):
    with open(file_to_write,mode='w') as f:
        for key,val in words_tags.items():
            first = ' '.join(key)
            f.write(f"{first}\t{val}\n")
        for key,val in d_signature.items():
            first = ' '.join(key)
            f.write(f"{first}\t{val}\n")


def list_tags(words_tags):
    tags = set(list(zip(*words_tags.keys()))[1])
    return tags


def q_mle_is_created(data, q_mle):
    # init
    tag_sentences = []
    for sent in data:
        tag_sent = []
        for word in sent:
            word_tag = word.split('/')
            tag_sent.append(word_tag[-1])
        tag_sentences.append(tag_sent)
    # make dictionary of the tags
    global q_mle_dict
    q_mle_dict = {}
    for sent in tag_sentences:
        for i, tag in enumerate(sent):
            if i == 0:  # First tag - Beginning of a sentence
                q_mle_dict[("START","START")] = q_mle_dict.get(("START","START"),0) + 1
                q_mle_dict[("START","START",tag)] = q_mle_dict.get(("START","START", tag), 0) + 1
                q_mle_dict[("START", tag)] = q_mle_dict.get(("START", tag), 0) + 1
                q_mle_dict["START"] = q_mle_dict.get("START",0) + 1
                q_mle_dict[tag] = q_mle_dict.get(tag, 0) + 1
                continue
            elif i == 1:  # Second tag
                q_mle_dict[("START", sent[0], tag)] = q_mle_dict.get(("START", sent[0], tag), 0) + 1
            else:  # For the third tag until the last tag
                q_mle_dict[(sent[i - 2], sent[i - 1], tag)] = q_mle_dict.get((sent[i - 2], sent[i - 1], tag), 0) + 1
            # For the second tag until the last tag
            q_mle_dict[(sent[i - 1], tag)] = q_mle_dict.get((sent[i - 1], tag), 0) + 1
            q_mle_dict[tag] = q_mle_dict.get(tag, 0) + 1
        # The end of sentence
        last_tag_idx = len(sent) - 1
        q_mle_dict[(sent[last_tag_idx - 1], sent[last_tag_idx], "END")] = q_mle_dict.get((sent[last_tag_idx - 1], sent[last_tag_idx], "END"), 0) + 1
        q_mle_dict[(sent[last_tag_idx], "END")] = q_mle_dict.get((sent[last_tag_idx], "END"), 0) + 1
    q_mle_dict["END"] = len(tag_sentences)

    # writing the dictionary to file name:
    with open(q_mle,mode='w') as f:
        for key, val in q_mle_dict.items():
            key = ' '.join(key) if type(key) == tuple else key
            f.write(f"{key}\t{val}\n")


def e_mle_is_created(data):
    # init
    words_tags = {}
    word_vocab = {}
    # making dictionary
    for sent in data:
        for i, word_tag in enumerate(sent):
            word_tag = word_tag.split('/')
            tag = word_tag[-1]
            word = '/'.join(word_tag[:-1])
            word_vocab[word] = word_vocab.get(word,0) + 1
            words_tags[(word, tag)] = words_tags.get((word, tag), 0) + 1

    #e_mle_dict = improve_e_mle(e_mle_dict)
    return word_vocab, words_tags


def find_word_rare(word_vocab):
    word_rare = set()
    for word, count in word_vocab.items():
        if count <= RARE_NUMBER:
            word_rare.add(word)
    return word_rare


def find_signature(word):
    #to complete
    #Number

    # if bool(re.search(r'^\d{2}$', word)):
    #     return '^twoDigitNumber'
    if bool(re.search(r'^\d{4}$', word)):
        return '^fourDigitNumber'
    if bool(re.search(r'^\d+\.?\d+$', word)):
        return '^Number'
    if bool(re.search(r'^(?:\d+,)+\d+\.?\d+$', word)):
        return '^commaAndNumber'
    if bool(re.search(r'^[0-9]+[0-9\-]+[0-9]+$', word)):
        return '^dashAndNumber'
    if bool(re.search(r'^[0-9]+[0-9/]+[0-9]+$', word)):
        return '^slashNumber'
    if bool(re.search(r'^\d{1,2}:\d{2}', word)):
        return '^hour'
    if bool(re.search(r'^[A-Z]+$', word)):
        return '^allCap'
    # if bool(re.search(r'^[A-Z]\.$',word)):
    #     return '^capPeriod'
    
    if bool(re.search(r'^[a-z]+ied$', word)):
        return '^suffix_ied'
    if bool(re.search(r'^[a-z]+ed$', word)):
        return '^suffix_ed'
    if bool(re.search(r'^[a-z]+ing$', word)):
        return '^suffix_ing'
    if bool(re.search(r'^[a-z]+tion$', word)):
        return '^suffix_tion'
    if bool(re.search(r'^[a-z]+sion$', word)):
        return '^suffix_sion'
    if bool(re.search(r'^[a-z]+able$', word)):
        return '^suffix_able'
    if bool(re.search(r'^[a-z]+ible$', word)):
        return '^suffix_ible'
    if bool(re.search(r'^[a-z]+ful$', word)):
        return '^suffix_ful'        
    if bool(re.search(r'^[a-z]+ence$', word)):  
        return '^suffix_ence'     
    # if bool(re.search(r'^[a-z]+sial$', word)): 
    #     return '^suffix_sial'      
    if bool(re.search(r'^[a-z]+tial$', word)): 
        return '^suffix_tial'      
    if bool(re.search(r'^[a-z]+ment$', word)):
        return '^suffix_ment'               
    if bool(re.search(r'^[a-z]+ly$', word)):
        return '^suffix_ly'                
    if bool(re.search(r'^[a-z]+est$', word)):
        return '^suffix_est'        
    if bool(re.search(r'^[a-z]+ian$', word)): 
        return '^suffix_ian'
    if bool(re.search(r'^[a-z]+ship$', word)): 
        return '^suffix_ship'     
    if bool(re.search(r'^[a-z]+ness$', word)): 
        return '^suffix_ness'     
    if bool(re.search(r'^[a-z]+hood$', word)):       
        return '^suffix_hood'
    if bool(re.search(r'^[a-z]+dom$', word)):
        return '^suffix_dom'        
    if bool(re.search(r'^[a-z]+ance$', word)):
        return '^suffix_ances'       
    if bool(re.search(r'^[a-z]+ist$', word)):
        return '^suffix_ist'        
    if bool(re.search(r'^[a-z]+ism$', word)):
        return '^suffix_ism'        
    if bool(re.search(r'^[a-z]+age$', word)):
        return '^suffix_age'        
    if bool(re.search(r'^[a-z]+er$', word)):
        return '^suffix_er'        
    if bool(re.search(r'^[a-z]+or$', word)):
        return '^suffix_or'         
    if bool(re.search(r'^[a-z]+ity$', word)): 
        return '^suffix_ity'       
    if bool(re.search(r'^[a-z]+ty$', word)): 
        return '^suffix_ty'               
    if bool(re.search(r'^[a-z]+ive$', word)):
        return '^suffix_ive'        
    if bool(re.search(r'^[a-z]+ish$', word)): 
        return '^suffix_ish'     
    if bool(re.search(r'^[a-z]+ize$', word)):
        return '^suffix_ize'        
    if bool(re.search(r'^[a-z]+ise$', word)):
        return '^suffix_ise'        
    if bool(re.search(r'^[a-z]+ify$', word)):
        return '^suffix_ify'        
    if bool(re.search(r'^[a-z]+ate$', word)):
        return '^suffix_ate'        
    if bool(re.search(r'^[a-z]+en$', word)):
        return '^suffix_en'        
    if bool(re.search(r'^[a-z]+ic$', word)):
        return '^suffix_ic'         
    if bool(re.search(r'^[a-z]+al$', word)):
        return '^suffix_al'               
    if bool(re.search(r'^[a-z]+less$', word)):
        return '^suffix_less'       
    if bool(re.search(r'^[a-z]+ous$', word)):
        return '^suffix_ous'   
    if bool(re.search(r'^[A-Z].*$' ,word)):
        return '^initCap' 
    return '^UNK'


def dict_signature(words_tags, word_rare):
    signatures_count = {}
    for word_tag, count in words_tags.items():
        word = word_tag[0]
        tag = word_tag[1]
        if word in word_rare:
            signature = find_signature(word)
            signatures_count[(signature,tag)] = signatures_count.get((signature,tag),0) + count
    return signatures_count


def read_data(input_file):
    with open(input_file, mode='r') as t:
        data = []
        for line in t.readlines():
            data.append(line.strip().split())
    return data


def main(input_file='data/ass1-tagger-train.txt', q_mle="q.mle.txt", e_mle="e.mle.txt"):
#def main(input_file='ner/train', q_mle="q_ner.mle.txt", e_mle="e_ner.mle.txt"):
    data = read_data(input_file)
    q_mle_is_created(data, q_mle)
    word_vocab, words_tags = e_mle_is_created(data)
    word_rare = find_word_rare(word_vocab)
    d_signature = dict_signature(words_tags, word_rare)
    writing_to_file(words_tags, d_signature, e_mle)


if __name__ == '__main__':
    input_file, q_mle, e_mle = sys.argv[1], sys.argv[2], sys.argv[3]
    main(input_file, q_mle, e_mle)
