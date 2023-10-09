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
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util


from sentence_transformers import SentenceTransformer, util

sw = set(stopwords.words())


FLAGS = flags.FLAGS

flags.DEFINE_integer("vocab_size", 1500, "")
flags.DEFINE_integer("window_size", 2, "")
flags.DEFINE_integer("edge_minimum_weight", 10, "")


flags.DEFINE_float("lamda", 0.7, "")
flags.DEFINE_float("per_word_probability", 0.01, "")
flags.DEFINE_float("relevance_weight", 0.6, "")

flags.DEFINE_string("pos_tags", "NN,VBG,RB,RBR,JJ,VB,VBD,VBN,NNS,NNP", "Pos tags to filter for")
flags.DEFINE_string("emotion", "anger", "Emotion to build pagerank for.")
flags.DEFINE_string("results_file", "results.csv", "")

flags.DEFINE_string("training_data_path", "", "")
flags.DEFINE_string("test_data_json", "", "")
flags.DEFINE_string("validation_data_json", "", "")

flags.DEFINE_boolean("force_embeddings_computation", False, "")

flags.DEFINE_string("embedding_directory", "", "")

def return_emo_word_probability():
  with open("NRC-Emotion-Intensity-Lexicon-v1.txt") as f:
    emotion_intensity = f.read().split('\n')

  emotion_word_probability = defaultdict(list)

  for w in emotion_intensity:
    l = w.split('\t')
    if len(l) == 3:
      emotion_word_probability[l[0]].append((l[1], l[2]))

  return emotion_word_probability


def filter_input(list_of_posts):
  for i in range(len(list_of_posts)):
    list_of_posts[i] = pos_tag(word_tokenize(list_of_posts[i].translate(str.maketrans('', '', string.punctuation))))
    for j in range(len(list_of_posts[i])):
      list_of_posts[i][j] = (list_of_posts[i][j][0].lower(), list_of_posts[i][j][1])
    list_of_posts[i] = [elem for elem in list_of_posts[i] if elem[0] not in sw]
  return list_of_posts

def build_graph_structure(list_of_posts, set_of_kept_words):
  word_probability = return_emo_word_probability()
  filtered_word_probability = {}
  for word in word_probability:
    for pair in word_probability[word]:
      if pair[0] == FLAGS.emotion:
        filtered_word_probability[word] = pair[1]

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

  frequency = defaultdict(int)
  for post in tqdm(list_of_posts):
    for i in range(len(post)):
      frequency[post[i][0]] += 1
      left = i - FLAGS.window_size
      right = i + FLAGS.window_size
      left = max(0, left)
      right = min(right, len(post))
      for j in range(left, right):
        if i != j:
          graph_structure[post[i][0]][post[j][0]] += 1 

  new_graph_structure = {}
  for elem in graph_structure:
    new_graph_structure[elem] = {}
    for e in graph_structure[elem]:
      if graph_structure[elem][e] >= FLAGS.edge_minimum_weight:
        new_graph_structure[elem][e] = graph_structure[elem][e] / frequency[e]

  graph_structure = new_graph_structure
  for elem in graph_structure:
    s = 0.0
    for e in graph_structure[elem]:
      s += graph_structure[elem][e]
    for e in graph_structure[elem]:
      graph_structure[elem][e] /= s

  return graph_structure, word_importance


def topical_pagerank(graph_structure, word_importance):

  relevance_mapping = {}
  for elem in graph_structure:
    relevance_mapping[elem] = 1.0 / len(graph_structure)

  total = len(graph_structure)

  previous_delta = -1
  for _ in tqdm(range(50)):
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
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
  m = 0
  r1, r2, rl = 0, 0, 0

  for e in test_data:
    scores = scorer.score(prediction[1], e[1])
    r1 = max(r1, scores['rouge1'].fmeasure)
    r2 = max(r2, scores['rouge2'].fmeasure)
    rl = max(rl, scores['rougeL'].fmeasure)

  return r1, r2, rl

def read_training_data():
  data_blob = pd.read_csv(FLAGS.training_data_path)
  return data_blob['selftext_cleaned'].tolist()

def stack_sentences_for_similarity(list_of_posts):
  stacked_sentences = []
  for post in list_of_posts:
    tokz = sent_tokenize(post)
    stacked_sentences += tokz

  return stacked_sentences

def read_and_write_embeddings(embedding_directory, stacked_training_sentences, stacked_validation_sentences, stacked_testing_sentences):

  model = SentenceTransformer('all-mpnet-base-v2', device="cuda")

  if not os.path.isdir(embedding_directory):
    os.mkdir(embedding_directory)

  training_embeddings_path = os.path.join(embedding_directory, 'embeddings_train.npy')
  validation_embeddings_path = os.path.join(embedding_directory, 'embeddings_validation.npy')
  testing_embeddings_path = os.path.join(embedding_directory, 'embeddings_test.npy')

  if os.path.isfile(training_embeddings_path) and FLAGS.force_embeddings_computation == False:
    with open(training_embeddings_path, 'rb') as f:
      embeddings_train = np.load(f)
  else:
    embeddings_train = model.encode(stacked_training_sentences, convert_to_tensor=True)
    with open(training_embeddings_path, 'wb') as f:
      np.save(f, embeddings_train.cpu())

  if os.path.isfile(validation_embeddings_path) and FLAGS.force_embeddings_computation == False:
    with open(validation_embeddings_path, 'rb') as f:
      embeddings_validation = np.load(f)
  else:
    embeddings_validation = model.encode(stacked_validation_sentences, convert_to_tensor=True)
    with open(validation_embeddings_path, 'wb') as f:
      np.save(f, embeddings_validation.cpu())


  if os.path.isfile(testing_embeddings_path) and FLAGS.force_embeddings_computation == False:
    with open(testing_embeddings_path, 'rb') as f:
      embeddings_test = np.load(f)
  else:
    embeddings_test = model.encode(stacked_testing_sentences, convert_to_tensor=True)
    with open(testing_embeddings_path, 'wb') as f:
      np.save(f, embeddings_test.cpu())

  return embeddings_train, embeddings_validation, embeddings_test

def read_test_data(path):

  with open(path) as f:
    test_data_json = json.load(f)

  test_data = test_data_json[FLAGS.emotion]

  list_of_test_posts = []
  for post in test_data:
    list_of_test_posts.append(post)

  return list_of_test_posts, test_data

def preprocess_data(list_of_posts, set_of_kept_words, freq_as_list):

  copy_of_data = deepcopy(list_of_posts)
  # Filter by pos_tag + lowercase + remove stopwords.
  set_of_pos_tags = set(FLAGS.pos_tags.split(','))
  copy_of_data = filter_input(copy_of_data)
  for i in range(len(copy_of_data)):
    copy_of_data[i] = [elem for elem in copy_of_data[i] if elem[1] in set_of_pos_tags]
  
  if not freq_as_list:
    # Filter by frequency
    frequency_dict = defaultdict(int)
    for post in copy_of_data:
      for elem in post:
        frequency_dict[elem[0]] += 1
    freq_as_list = [(frequency_dict[k], k) for k in frequency_dict]
    if len(freq_as_list) > FLAGS.vocab_size:
      freq_as_list.sort(reverse=True)
      freq_as_list = freq_as_list[:FLAGS.vocab_size]
  
  if not set_of_kept_words:
    set_of_kept_words = set([elem[1] for elem in freq_as_list])

  for i in range(len(copy_of_data)):
    copy_of_data[i] = [elem for elem in copy_of_data[i] if elem[0] in set_of_kept_words]

  return copy_of_data, set_of_kept_words, freq_as_list

def normalize_relevance(relevance):
  maximum = -1
  minimum = 100
  for elem in relevance:
    if relevance[elem] > maximum:
      maximum = relevance[elem]
    
    if relevance[elem] < minimum:
      minimum = relevance[elem]

  for elem in relevance:
      relevance[elem] = (relevance[elem] - minimum) / (maximum - minimum)

def compute_relevance_scores(preprocessed_stacked_train_data, set_of_kept_words, relevance):
  train_relevance_scores = []
  for sentence in tqdm(preprocessed_stacked_train_data):
    num_words = 0
    crt_relevance = 0.0
    for word in sentence:
      if word[0] in set_of_kept_words:
        num_words += 1
        crt_relevance += relevance[word[0]]
    if crt_relevance == 0:
      train_relevance_scores.append(0)
    else:
      train_relevance_scores.append(crt_relevance / num_words)
  
  return train_relevance_scores

def compute_meaning_scores(embeddings_train, embeddings_test, train_relevance_scores, test_relevance_scores):

  cosine_scores = util.cos_sim(embeddings_test, embeddings_train)
  numpy_similarity = cosine_scores.cpu().numpy()

  test_meaning_scores = []
  for i in range(len(test_relevance_scores)):
    test_meaning_scores.append(np.mean(numpy_similarity[i, :] * np.array(train_relevance_scores)))

  return test_meaning_scores

def compute_post_summaries(list_of_test_posts, final_sentence_scores, threshold):
  sentence_idx = 0
  list_of_summaries = []
  list_of_summary_sentence_idx = []
  lengths = []

  for post in list_of_test_posts:
    summary = []
    summary_sentence_idx = []
    sentences = sent_tokenize(post)
    for i in range(len(sentences)):
      if final_sentence_scores[sentence_idx] > threshold:
        summary.append(sentences[i])
        summary_sentence_idx.append(i)
      sentence_idx += 1

    list_of_summaries.append(" ".join(summary))
    list_of_summary_sentence_idx.append(summary_sentence_idx)
    lengths.append(len(sentences))


  assert (len(list_of_summaries)) == len(list_of_test_posts)
  assert sentence_idx == len(final_sentence_scores)

  return list_of_summaries, list_of_summary_sentence_idx, lengths

def evaluate_summaries(summaries, list_of_summary_sentence_idx, lengths, test_data):

  assert len(summaries) == len(list_of_summary_sentence_idx) and len(list_of_summary_sentence_idx) == len(lengths)

  topic_predictions = {}
  avg_r1 = 0
  avg_r2 = 0
  avg_rl = 0
  tpt = 0
  fpt = 0
  tnt = 0
  fnt = 0

  idx = 0
  for post in test_data:
    r1, r2, rl = evaluate((list_of_summary_sentence_idx[idx], summaries[idx], lengths[idx]), test_data[post])
    avg_r1 += r1 / len(test_data)
    avg_r2 += r2 / len(test_data)
    avg_rl += rl / len(test_data)

    idx += 1


  return avg_r1, avg_r2, avg_rl

def main(argv):

  list_of_train_posts = read_training_data()
  stacked_train_sentences = stack_sentences_for_similarity(list_of_train_posts)

  list_of_validation_posts, validation_data = read_test_data(FLAGS.validation_data_json)
  stacked_validation_sentences = stack_sentences_for_similarity(list_of_validation_posts)

  list_of_test_posts, test_data = read_test_data(FLAGS.test_data_json)
  stacked_test_sentences = stack_sentences_for_similarity(list_of_test_posts)

  embeddings_train, embeddings_validation, embeddings_test = read_and_write_embeddings(FLAGS.embedding_directory, stacked_train_sentences, stacked_validation_sentences, stacked_test_sentences)

  preprocessed_train_data, set_of_kept_words, freq_as_list = preprocess_data(list_of_train_posts, None, None)
  preprocessed_stacked_train_data, _, _ = preprocess_data(stacked_train_sentences, set_of_kept_words, freq_as_list)

  preprocessed_validation_data, _, _ = preprocess_data(list_of_validation_posts, set_of_kept_words, freq_as_list)
  preprocessed_stacked_validation_data, _, _ = preprocess_data(stacked_validation_sentences, set_of_kept_words, freq_as_list)

  preprocessed_test_data, _, _ = preprocess_data(list_of_test_posts, set_of_kept_words, freq_as_list)
  preprocessed_stacked_test_data, _, _ = preprocess_data(stacked_test_sentences, set_of_kept_words, freq_as_list)

  graph_structure, word_importance = build_graph_structure(preprocessed_train_data, set_of_kept_words)
  relevance = topical_pagerank(graph_structure, word_importance)

  normalize_relevance(relevance)

  train_relevance_scores = compute_relevance_scores(preprocessed_stacked_train_data, set_of_kept_words, relevance)
  validation_relevance_scores = compute_relevance_scores(preprocessed_stacked_validation_data, set_of_kept_words, relevance)
  test_relevance_scores = compute_relevance_scores(preprocessed_stacked_test_data, set_of_kept_words, relevance)

  validation_meaning_scores = compute_meaning_scores(embeddings_train, embeddings_validation, train_relevance_scores, validation_relevance_scores)
  test_meaning_scores = compute_meaning_scores(embeddings_train, embeddings_test, train_relevance_scores, test_relevance_scores)

  final_sentence_scores_validation = []
  for i in range(len(validation_meaning_scores)):
    final_sentence_scores_validation.append(validation_meaning_scores[i] * (1 - FLAGS.relevance_weight) + validation_relevance_scores[i] * FLAGS.relevance_weight)

  final_sentence_scores_test = []
  for i in range(len(test_meaning_scores)):
    final_sentence_scores_test.append(test_meaning_scores[i] * (1 - FLAGS.relevance_weight) + test_relevance_scores[i] * FLAGS.relevance_weight)

  copy_of_cal_sentences = deepcopy(final_sentence_scores_validation)
  copy_of_cal_sentences.sort()

  thresholds_to_test = []
  for i in range(100):
    thresholds_to_test.append(copy_of_cal_sentences[(len(copy_of_cal_sentences) * i) // 100])

  crt_threshold = -1
  crt_performance = -1
  for threshold in thresholds_to_test:
    print('TESTING THRESHOLD: ', threshold)
    summaries, list_of_summary_sentence_idx, lengths = compute_post_summaries(list_of_validation_posts, final_sentence_scores_validation, threshold)
    avg_r1, avg_r2, avg_rl = evaluate_summaries(summaries, list_of_summary_sentence_idx, lengths, validation_data)
    if avg_r2 > crt_performance:
      crt_performance = avg_r2
      crt_threshold = threshold
    df = pd.read_csv('validation_results.csv')
    df.loc[len(df)] = [avg_r1, avg_r2, avg_rl, FLAGS.vocab_size, FLAGS.window_size, FLAGS.edge_minimum_weight, FLAGS.lamda, FLAGS.per_word_probability, FLAGS.relevance_weight, threshold, FLAGS.emotion]
    df.to_csv('validation_results.csv', index=False)
  
  summaries, list_of_summary_sentence_idx, lengths = compute_post_summaries(list_of_test_posts, final_sentence_scores_test, crt_threshold)
  avg_r1, avg_r2, avg_rl = evaluate_summaries(summaries, list_of_summary_sentence_idx, lengths, test_data)
  df = pd.read_csv('test_results.csv')
  df.loc[len(df)] = [avg_r1, avg_r2, avg_rl, FLAGS.vocab_size, FLAGS.window_size, FLAGS.edge_minimum_weight, FLAGS.lamda, FLAGS.per_word_probability, FLAGS.relevance_weight, crt_threshold, FLAGS.emotion]
  df.to_csv("test_results.csv", index=False)


if __name__ == "__main__":
  app.run(main)