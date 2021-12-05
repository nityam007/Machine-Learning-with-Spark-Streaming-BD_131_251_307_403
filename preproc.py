from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger


stemmer = PorterStemmer()
tagger = PerceptronTagger()
lemmatizer = WordNetLemmatizer()


def remove_stop_words(str):
    stop_words = set(stopwords.words("english"))
    words = str.split()
    clean_str = " ".join([word for word in words if not word in stop_words])
    return clean_str



def lemmatize(str):
    words = str.split()
    tagged_words = tagger.tag(words)
    clean_str = []

    for word in tagged_words:
        if 'n' in word[1].lower():
            lemma = lemmatizer.lemmatize(word[0], pos='n')
        else:
            lemma = lemmatizer.lemmatize(word[0], pos='v')

        clean_str.append(lemma)

    return " ".join(clean_str)


def preproc_pipeline(str):
    rm_stop = remove_stop_words(str)
    lem = lemmatize(rm_stop)
    return lem