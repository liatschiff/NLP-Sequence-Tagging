import sys
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

appeared_words = set()


def training_model(y, x):
    v = DictVectorizer()
    x_vectors = v.fit_transform(x)
    with open('feature_map_file', 'wb') as m:
        pickle.dump(v, m)
        pickle.dump(appeared_words, m)
    clf = LogisticRegression(multi_class='multinomial', penalty='l2', tol=1e-4, random_state=0, n_jobs=4, max_iter=1555)
    clf.fit(x_vectors, y)
    return clf


def read_features_file(file):
    global appeared_words
    with open(file, mode='r') as f:
        x = []
        y = []
        for line in f.readlines():
            x.append(dict())
            for f_word in line.split():
                index_eq = f_word.rindex('=')
                key = (f_word)[:index_eq]
                val = (f_word)[index_eq + 1:]
                if key == "label":
                    y.append(val)
                else:
                    x[-1][key] = val
                    if key == "form":
                        appeared_words.add(val)
    return y, x


def main(features_file="features_file.txt", model_file="model_file"):
    y, x = read_features_file(features_file)
    clf = training_model(y, x)
    f = open(model_file, 'wb')
    pickle.dump(clf, f)
    f.close()


if __name__ == '__main__':
    features_file, model_file = sys.argv[1], sys.argv[2]
    main(features_file, model_file)
