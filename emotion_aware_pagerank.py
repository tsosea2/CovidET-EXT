from turtle import update
import json
import nltk
import numpy as np
import os
import pandas as pd
import string

from absl import app
from absl import flags
from copy import deepcopy
from cgitb import text
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from rouge_score import rouge_scorer
from tqdm import tqdm



nltk.download('stopwords')
sw = set(stopwords.words())


FLAGS = flags.FLAGS

flags.DEFINE_integer("vocab_size", 1000, "")
flags.DEFINE_integer("window_size", 5, "")

flags.DEFINE_float("lamda", 0.6, "")
flags.DEFINE_float("per_word_probability", 0.05, "")
flags.DEFINE_float("relevance_threshold", 0.4, "")

flags.DEFINE_string("pos_tags", "NN,VBG,RB,RBR,JJ,VB,VBD,VBN,NNS,NNP", "Pos tags to filter for")
flags.DEFINE_string("emotion", "joy", "Emotion to build pagerank for.")
flags.DEFINE_string("results_file", "results.csv", "")


def softmax(x, temperature=1):
    return np.exp(x/temperature)/sum(np.exp(x/temperature))

def return_emo_word_probability():
  with open("NRC-Emotion-Intensity-Lexicon-v1.txt") as f:
    emotion_intensity = f.read().split('\n')

  emotion_word_probability = defaultdict(list)

  for w in emotion_intensity:
    l = w.split('\t')
    if len(l) == 3:
      emotion_word_probability[l[0]].append((l[1], l[2]))

  return emotion_word_probability

from nltk import sent_tokenize

def filter_input(list_of_posts, set_of_pos_tags):

  for i in range(len(list_of_posts)):
    list_of_posts[i] = pos_tag(word_tokenize(list_of_posts[i].translate(str.maketrans('', '', string.punctuation))))
    for j in range(len(list_of_posts[i])):
      list_of_posts[i][j] = (list_of_posts[i][j][0].lower(), list_of_posts[i][j][1])
    list_of_posts[i] = [elem for elem in list_of_posts[i] if elem[0] not in sw]
  return list_of_posts

def build_graph_structure(list_of_posts, graph_structure):
  for post in tqdm(list_of_posts):
    for i in range(len(post)):
      left = i - FLAGS.window_size
      right = i + FLAGS.window_size
      left = max(0, left)
      right = min(right, len(post))
      for j in range(left, right):
        if i != j:
          graph_structure[post[i][0]][post[j][0]] += 1 

  for elem in graph_structure:
    s = 0.0
    for e in graph_structure[elem]:
      s += graph_structure[elem][e]
    for e in graph_structure[elem]:
      graph_structure[elem][e] /= s

  return graph_structure


def topical_pagerank(relevance_mapping, graph_structure, word_importance):

  total = len(graph_structure)

  previous_delta = -1
  for _ in range(50):
    relevance_mapping_aux = deepcopy(relevance_mapping)
    total_delta = 0
    for w in graph_structure:
      new_relevance = 0
      for near_word in graph_structure[w]:
        new_relevance += FLAGS.lamda * graph_structure[w][near_word] * relevance_mapping_aux[near_word]
      new_relevance += (1 - FLAGS.lamda) * word_importance[w]
      delta = abs((relevance_mapping_aux[w] - new_relevance)) / total
      relevance_mapping[w] = new_relevance
      total_delta += delta
    if previous_delta == total_delta:
      break
    previous_delta = total_delta

  return relevance_mapping



def evaluate(prediction, test_data):

  sents_pred = set(prediction[0])
  sents_actual = []
  for e in test_data:
    sents_actual += e[0]
  sents_actual = set(sents_actual)

  tp = len(sents_actual.intersection(sents_pred))
  fp = len(sents_pred.difference(sents_actual))
  fn = len(sents_actual.difference(sents_pred))
  tn = prediction[2] - len(sents_pred.union(sents_actual))

  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
  m = 0
  r1, r2, rl = 0, 0, 0
  pred = " ".join(prediction[1])
  for e in test_data:
    scores = scorer.score(pred, e[1])
    r1 = max(r1, scores['rouge1'].fmeasure)
    r2 = max(r2, scores['rouge2'].fmeasure)
    rl = max(rl, scores['rougeL'].fmeasure)

  return tp, fp, tn, fn,  r1, r2, rl

def main(argv):

  set_of_pos_tags = set(FLAGS.pos_tags.split(','))

  word_probability = return_emo_word_probability()
  filtered_word_probability = {}
  for word in word_probability:
    for pair in word_probability[word]:
      if pair[0] == FLAGS.emotion:
        filtered_word_probability[word] = pair[1]

  data_blob = pd.read_csv('support_data_filtered.csv')
  list_of_posts = data_blob['selftext_cleaned'].tolist()
  list_of_posts = filter_input(list_of_posts, set_of_pos_tags)
  for i in range(len(list_of_posts)):
    list_of_posts[i] = [elem for elem in list_of_posts[i] if elem[1] in set_of_pos_tags]

  frequency_dict = defaultdict(int)

  for post in list_of_posts:
    for elem in post:
      frequency_dict[elem[0]] += 1

  freq_as_list = [(frequency_dict[k], k) for k in frequency_dict]
  if len(freq_as_list) > FLAGS.vocab_size:
    freq_as_list.sort(reverse=True)
    freq_as_list = freq_as_list[:FLAGS.vocab_size]
    set_of_kept_words = set([elem[1] for elem in freq_as_list])

  for i in range(len(list_of_posts)):
    list_of_posts[i] = [elem for elem in list_of_posts[i] if elem[0] in set_of_kept_words]

  word_importance = {}
  s = 0
  for elem in set_of_kept_words:
    if elem in filtered_word_probability:
      word_importance[elem] = float(filtered_word_probability[elem])
    else:
      word_importance[elem] = FLAGS.per_word_probability

    s +=  word_importance[elem]

  for elem in word_importance:
    word_importance[elem] /= s

  graph_structure = {}
  for elem in word_importance:
    graph_structure[elem] = defaultdict(float)
  relevance_mapping = {}
  build_graph_structure(list_of_posts, graph_structure)
  for elem in graph_structure:
    relevance_mapping[elem] = 1.0 / len(graph_structure)

  relevance = topical_pagerank(relevance_mapping, graph_structure, word_importance)

  for elem in relevance:
    relevance[elem] *= len(relevance_mapping)

  x = np.array([relevance[k] for k in relevance])
  mean_relevance = np.mean(x)
  crt_threshold = mean_relevance + FLAGS.relevance_threshold

  topic_predictions = {}
  with open("test_proc.json") as f:
    test_data = json.load(f)
  
  test_data = test_data[FLAGS.emotion]
  avg_r1 = 0
  avg_r2 = 0
  avg_rl = 0
  tpt = 0
  fpt = 0
  tnt = 0
  fnt = 0
  for post in test_data:
    tp = []
    actual_sentences = []
    l = sent_tokenize(post)
    opt = filter_input(deepcopy(l), set_of_pos_tags)
    for i in range(len(opt)):
      opt[i] = [elem[0] for elem in opt[i] if elem[1] in set_of_pos_tags and elem[0] in graph_structure]
    for i in range(len(opt)):
      if len(opt[i]) > 0:
        opt[i] = sum([relevance[elem] for elem in opt[i]]) / len(opt[i])
      else:
        opt[i] = 0

      if opt[i] > crt_threshold:
        tp.append(i)
        actual_sentences.append(l[i])

    topic_predictions[post] = (tp, actual_sentences, len(opt))


    tp, fp, tn, fn, r1, r2, rl = evaluate(topic_predictions[post], test_data[post])

    tpt += tp
    fpt += fp 
    tnt += tn
    fnt += fn 
    avg_r1 += r1 / len(test_data)
    avg_r2 += r2 / len(test_data)
    avg_rl += rl / len(test_data)

  
  precision = tpt / (tpt + fpt)
  recall = tpt / (tpt + fnt)
  df = pd.read_csv('results.csv')
  df.loc[len(df)] = [2 * precision * recall / (precision + recall), avg_r1, avg_r2, avg_rl, FLAGS.vocab_size, FLAGS.window_size, FLAGS.lamda, FLAGS.per_word_probability, FLAGS.relevance_threshold, FLAGS.emotion]
  df.to_csv("results.csv", index=False)


if __name__ == "__main__":
  app.run(main)