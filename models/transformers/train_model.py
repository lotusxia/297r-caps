import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow_text as text
import tensorflow_hub as hub

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

import tensorflow_addons as tfa

data = "./data"

print("Loading data...")

with open(f"{data}/clean/test_dict.pickle", "rb") as f:
    test_dict = pickle.load(f)

with open(f"{data}/clean/train_dict.pickle", "rb") as f:
    train_dict = pickle.load(f)

train_df = pd.read_pickle(f"{data}/clean/train_df.pickle")
test_df = pd.read_pickle(f"{data}/clean/test_df.pickle")

# Model Parameters
MAX_SEQ_LENGTH = 60
LATENT_LAYER_SIZE = 10
READING_IN_CHUNKS = 100
TRAIN_PROPORTION = 1  # change later
TRAIN_EPOCHS = 5
ACTIVATION_ALPHA = 0.04
BATCH_SIZE = 32
DROPOUT_RATE = 0.1
EPOCH_STEPS = 10000

def review_text_generator(review_dict=train_dict, 
                          chunk_size=BATCH_SIZE):
    """function (generator) to read review text piece by piece."""
    chunk_number = 0
    while True:
        if chunk_number+chunk_size >= len(review_dict['asin']):
            asin = review_dict['asin'][chunk_number:]
            month = review_dict['year_month'][chunk_number:]
            rank_feature = review_dict['rolling_median_month_rank'][chunk_number:]
            target = review_dict['target_sales'][chunk_number:]
            yield asin, month, rank_feature, target
            break
        asin = review_dict['asin'][chunk_number:chunk_number+chunk_size]
        month = review_dict['year_month'][chunk_number:chunk_number+chunk_size]
        rank_feature = review_dict['rolling_median_month_rank'][chunk_number:chunk_number+chunk_size]
        target = review_dict['target_sales'][chunk_number:chunk_number+chunk_size]
        chunk_number += chunk_size
        yield asin, month, rank_feature, target

def train_generator(
    text_generator, review_dict=train_dict, products_df=train_df,
    number_of_chunks=READING_IN_CHUNKS, batch_size=1, uncased=False
    ):
    """function (generator) to create cumulative data one by one."""
    # Note: This generator will keep repeating, so stop condition must be in
    # the iterator or fit method
    while True: 
      for asins, months, rank_feature, sales in text_generator(review_dict,
                                                              number_of_chunks):
        text_batch, rank_feats_batch, sales_batch = [], [], []
        for i in range(batch_size):
          # create cumulative data
          if len(rank_feature[i]) != 30: continue
          filtered_data = products_df[(products_df['asin']==asins[i]) & 
                                      (products_df['year_month']<=months[i])]
          text = np.concatenate(list(filtered_data['review_text']))
          if uncased: text = np.array([review_text.lower() for review_text in text])
          text = np.expand_dims(text, 0)
          # vote = np.ones(shape=np.concatenate(list(filtered_data['reviewvotes_num'])).shape)
          # if not len(text)==len(vote) or np.all(vote==0): continue
          rank_feats = np.array([rank_feature[i]]) #np.vstack((np.array([rank_feature[i]]), np.zeros((len(text)-1, 30))))
          sale = np.expand_dims(np.array(sales[i]), 0)
          text_batch.append(text), rank_feats_batch.append(rank_feats), sales_batch.append(sale)
        yield [text_batch, rank_feats_batch], sales_batch

def train_generator_textonly(
    text_generator, review_dict=train_dict, products_df=train_df,
    number_of_chunks=READING_IN_CHUNKS, batch_size=1, uncased=False
    ):
    """function (generator) to create cumulative data one by one."""
    # Note: This generator will keep repeating, so stop condition must be in
    # the iterator or fit method
    while True: 
      for asins, months, rank_feature, sales in text_generator(review_dict,
                                                              number_of_chunks):
        text_batch, rank_feats_batch, sales_batch = [], [], []
        for i in range(batch_size):
          # create cumulative data
          if len(rank_feature[i]) != 30: continue
          filtered_data = products_df[(products_df['asin']==asins[i]) & 
                                      (products_df['year_month']<=months[i])]
          text = np.concatenate(list(filtered_data['review_text']))
          if uncased: text = np.array([review_text.lower() for review_text in text])
          text = np.expand_dims(text, 0)
          # vote = np.ones(shape=np.concatenate(list(filtered_data['reviewvotes_num'])).shape)
          # if not len(text)==len(vote) or np.all(vote==0): continue
          rank_feats = np.array([rank_feature[i]]) #np.vstack((np.array([rank_feature[i]]), np.zeros((len(text)-1, 30))))
          sale = np.expand_dims(np.array(sales[i]), 0)
          text_batch.append(text), rank_feats_batch.append(rank_feats), sales_batch.append(sale)
        yield [text_batch], sales_batch

print("Loading model...")
model = tf.keras.models.load_model('trained_model_sales_linear2')

training_history = model.fit(
    train_generator_textonly(
        review_text_generator,
        batch_size=1,
        uncased=True), 
    epochs=TRAIN_EPOCHS,
    steps_per_epoch=EPOCH_STEPS)

model.save('trained_model_sales_linear3')
