import warnings
warnings.filterwarnings('ignore')
import copy
import contractions
import nltk
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy
import re
import string
from contractions import contractions_dict
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stopword_list = nltk.corpus.stopwords.words('english')
PUNCT_TO_REMOVE = string.punctuation
bracket_expression="([\(\[]).*?([\)\]])"
EMAIL_REGEX = r'[a-zA-Z\.-]+@[\w\.-]+'
URL_REGEX = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
SPECIAL_CHARS = r'[^a-zA-z0-9.,!?/:;\"\'\s]'#
nlp = spacy.load('en')

#function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans(' ', ' ', PUNCT_TO_REMOVE))

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
#function to expand contractions
def expand_contractions(s, contractions_dict=contractions_dict):
     result=s
     def replace(match):
         return contractions_dict[match.group(0)]
     try:
        result=contractions_re.sub(replace, s)
     except:
        pass   
     return result

#function to remove non-ascii characters
def remove_non_Ascii(s): return "".join(i for i in s if ord(i)<128)

#get stop words of different languages
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}
#detect language based on number of stop words for given language
def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    lang = max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]
    if lang == 'english':
        return True
    else:
        return False

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatize_text(text):
    text = nlp(text, disable=['parser','ner'])
    text = [word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text]
    return text  


def prepare_data(df,col):
  new_col="processed_"+col
  # convert to str type
  df[new_col]=df[col].apply(lambda x:str(x))
  # drop duplicated lines
  df=df.drop_duplicates(keep='first')
  # remove non-ascii characters
  df[new_col]=df[new_col].apply(lambda x:remove_non_Ascii(x))
  # remove e-mails
  df[new_col]=df[new_col].apply(lambda x: re.sub(EMAIL_REGEX,"email",x))
  # remove urls
  df[new_col]=df[new_col].apply(lambda x: re.sub(URL_REGEX, "url",x) )
  # remove all in brackets
  df[new_col]=df[new_col].apply(lambda x: re.sub(bracket_expression, "\g<1>\g<2>", x) )
  # strip HTML  
  df[new_col]=df[new_col].apply(lambda x:strip_html_tags(x))
  # remove extra newlines
  df[new_col]=df[new_col].apply(lambda x: re.sub(r'[\r|\n|\r\n]+', ' ',x))
  df[new_col]=df[new_col].apply(lambda x: x.replace('-',' ') )
  # keep only english comments
  df["eng"]=df[new_col].apply(lambda x:get_language(x))
  df=df[df["eng"]==True]
  df=df[[col,new_col]]
  # expand contractions
  df[new_col]=df[new_col].apply(lambda x: expand_contractions(str(x), contractions_dict=contractions_dict) )  
  # remove special characters
  df[new_col]=df[new_col].apply(lambda x: re.sub(SPECIAL_CHARS, '', x) )
  df[new_col]=df[new_col].apply(lambda x:remove_punctuation(x))
  df[new_col]=df[new_col].apply(lambda x:' '.join(x.split()))
  # remove extra spaces
  df[new_col]=df[new_col].apply(lambda x: re.sub(' +', ' ', x) )
  # lemmatize
  df[new_col]=df[new_col].apply(lambda x:lemmatize_text(x.lower()))  
  return df.reset_index(drop=True)


#function to filter ADJ/NN bigrams
def filter_matching(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in stopword_list or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[1][1] in second_type:
        return True
    else:
        return False                