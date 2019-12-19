import emoji
import re
import string
import spacy
from nltk.corpus import stopwords


def clean_emoji(text):
  if (emoji.emoji_count(text) > 0):
    return re.sub(emoji.get_emoji_regexp(), r"", text)
  else:
    return text

def clean_punctuation(text):
  """
  @text: is a text with/without the punctuation
  @return a text without punctuation
  """
  re_punc = re.compile("[%s]" % re.escape(string.punctuation))
  return re_punc.sub("", text)

def clean_at(text):
  re_at = r"^@\S+"
  return re.sub(re_at, "", text)

# pip install -U spacy && python -m spacy download en
nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
SW =  set(stopwords.words("english"))
SW.add("askebay")

def normalize(text):

  """
  Lemmatize tokens and remove them if they are stop words
  """
  text = nlp(text.lower())
  lemmatized = list()
  for word in text:
    lemma = word.lemma_.strip()
    if (lemma and lemma not in SW ):
      lemmatized.append(lemma)
  return lemmatized

def decontracted(text):
  # specific
  text = re.sub(r"won’t", "will not", text)
  text = re.sub(r"won\'t", "will not", text)

  text = re.sub(r"can’t", "can not", text)
  text = re.sub(r"can\'t", "can not", text)

  text = re.sub(r"dont’t", "do not", text)
  text = re.sub(r"don\'t", "do not", text)

  text = re.sub(r"doesn’t", "does not", text)
  text = re.sub(r"doesn\'t", "does not", text)

  text = re.sub(r"y’all", "you all", text)
  text = re.sub(r"y\'all", "you all", text)
  # general
  text = re.sub(r"’t", " not", text)
  text = re.sub(r"n\'t", " not", text)

  text = re.sub(r"’re", " are", text)
  text = re.sub(r"\'re", " are", text)

  text = re.sub(r"’s", " is", text)
  text = re.sub(r"\'s", " is", text)

  text = re.sub(r"’d", " would", text)
  text = re.sub(r"\'d", " would", text)

  text = re.sub(r"’ll", " will", text)
  text = re.sub(r"\'ll", " will", text)

  text = re.sub(r"’t", " not", text)
  text = re.sub(r"\'t", " not", text)

  text = re.sub(r"’ve", " have", text)
  text = re.sub(r"\'ve", " have", text)

  text = re.sub(r"’m", " am", text)
  text = re.sub(r"\'m", " am", text)
  return text

def clean_num(text):
  """
  @Remove numberics in the text
  @return the text without numberics
  """
  re_num = r"\d+"
  return re.sub(re_num, "", text)

def preprocessing(text):
#  text = clean_at(text)
  text = clean_emoji(text)
  text = decontracted(text)
  text = clean_punctuation(text)
  text = clean_num(text)
  text = normalize(text)
  return text


if __name__ == "__main__":
    
  #  import spacy
  #  nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
  print("I'm NLP helper")
  #spacy.load()
  # stops = spacy.lang.en.stop_words.STOP_WORDS
  # print(stop)
  ''''
  text = ["@applesupport tried resetting my settings .. restarting my phone .. all that"
        , "i need answers because it is annoying 🙃"
        , "@115855 That's great it has iOS 11.1 as we can rule out being outdated. Any steps tried since this started?"]
  print( [clean_at(sent) for sent in text] )
  '''



