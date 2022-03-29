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

# input folders
data = "./data"

# Loading the review texts
review_df = pd.read_pickle(f'{data}/clean/month_level_review.pickle')
review_df = review_df[['asin', 'year_month', 'review_text', 'reviewvotes_num']]

# Loading the rank data
bsr_df = pd.read_pickle(f'{data}/clean/month_level_rank_sales_price.pickle')
bsr_df = bsr_df[[
  'asin', 'year_month', 'mean_month_rank', 'median_month_rank',
  'rolling_median_month_rank', 'mean_month_est_sales', 'median_month_est_sales']]
print(bsr_df[["median_month_est_sales"]].describe())
# processing rank df for the merge
bsr_df['year_month'] = pd.to_datetime(bsr_df['year_month'])
bsr_df = bsr_df.sort_values(['asin', 'year_month'])
bsr_df['target_sales'] = bsr_df.groupby(['asin'])["median_month_est_sales"].shift(-1)

# processing review df for the merge
review_df['year_month'] = pd.to_datetime(review_df['year_month'])

# merging the data sets
products_df = pd.merge(review_df, bsr_df, on=["asin", "year_month"])
products_df = products_df.dropna()

with open(f"{data}/clean/product_sample.pickle", "rb") as f:
    product_sample = pickle.load(f)

train_asins = set(product_sample['train'])
test_asins = set(product_sample['test'])

# creating the target variable
#products_df['rank_change'] = products_df['target_rank'] - products_df['median_month_rank']

def process_moving_median_months(row):
  """Processes the moving median column to make sure it
  doesn't break our model"""
  rolling_median = np.array(row['rolling_median_month_rank'])
  # impute the NaNs
  try: rolling_median[np.isnan(rolling_median)] = np.nanmean(rolling_median)
  except: return np.ones(30)*0.24191888901999709
  # make sure all are of length 30 (prioritising the last 30 values)
  if len(rolling_median) < 30:
    impute_value = np.nanmean(rolling_median)
    rolling_median = np.append(np.array([impute_value]*(30-len(rolling_median))), rolling_median)
  else:
    rolling_median = rolling_median[-30:]
  return rolling_median

products_df['rolling_median_month_rank'] = products_df.apply(process_moving_median_months, axis=1)

def create_num_reviews(row):
  try: return len(row["review_text"])
  except: return 0

products_df["num_reviews"] = products_df.apply(create_num_reviews, axis=1)

print("Data created...")
print(products_df.head())

print("DF size before filtering:", len(products_df))
train_df = products_df[products_df['asin'].isin(train_asins)]
test_df = products_df[products_df['asin'].isin(test_asins)]
print("train size after filtering: ", len(train_df))
print("test size after filtering: ", len(test_df))

# Freeing up RAM
del bsr_df
del review_df
del products_df

# changing data to series

train_dict = train_df.to_dict('series')
for key in train_dict.keys():
  train_dict[key] = list(train_dict[key])

train_dict['review_text'] = [list(map(str, reviews_text_list)) for reviews_text_list in train_dict['review_text']]

test_dict = test_df.to_dict('series')
for key in test_dict.keys():
  test_dict[key] = list(test_dict[key])

test_dict['review_text'] = [list(map(str, reviews_text_list)) for reviews_text_list in test_dict['review_text']]

# Model Parameters
MAX_SEQ_LENGTH = 60
LATENT_LAYER_SIZE = 10
READING_IN_CHUNKS = 100
TRAIN_PROPORTION = 1  # change later
TRAIN_EPOCHS = 1
ACTIVATION_ALPHA = 0.04
BATCH_SIZE = 32
DROPOUT_RATE = 0.1
EPOCH_STEPS = 100

# resizing data based on TRAIN_PROPORTION
for key in train_dict.keys():
  train_dict[key] = train_dict[key][:int(len(train_dict[key])*TRAIN_PROPORTION)]

print(list(train_dict.keys()))

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

# confirming that the generator works as expected
for [text], sale in train_generator_textonly(review_text_generator,
                                               uncased=True,
                                               batch_size=32):
  print("Text:", len(text))
  print("Sale:", len(sale))
  break


def create_bert(max_seq_length=MAX_SEQ_LENGTH):
  """Returns BERT encodings."""
  # takes in text of the form np.array(["first review.", ...])
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  # creates BERT tokens for the text
  preprocessor = hub.KerasLayer(
      "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
  encoder_inputs = preprocessor(text_input)
  # encodes the input tokens
  encoder = hub.KerasLayer(
      "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2",
      trainable=False
  )
  # retrieves the vector representation of all the reviews seperately
  encodings = encoder(encoder_inputs)["pooled_output"]
  # creating final model
  model = tf.keras.models.Model(inputs=text_input, outputs=encodings,
                                name="BERT_uncased")
  return model

def create_model(transformer, latent_size=LATENT_LAYER_SIZE,
                 max_seq_length=MAX_SEQ_LENGTH,
                 activation_alpha=ACTIVATION_ALPHA,
                 dropout_rate=DROPOUT_RATE):
  """Returns model structure for regression on transformer embeddings."""
  # Input text of the form: np.array(["first review.", ...])
  text_input = tf.keras.layers.Input(shape=(None,), dtype=tf.string,
                                     name="text_input")
  # previous month bsr timeline
  previous_bsr_input = tf.keras.layers.Input(shape=(30,), dtype=tf.float32,
                                      name="bsr_input")
  # create encodings for the text
  encodings = transformer(text_input[0])
  # add additional dense layer to interpret transformer encodings and keep
  # shapes constant regardless of transformer shape
  latent_description = tf.keras.layers.Dense(
      latent_size, activation=tf.keras.layers.LeakyReLU(alpha=activation_alpha),
      name="latent_desc")(encodings)
  latent_description = tf.keras.layers.Dropout(dropout_rate)(latent_description)
  # weigh the reshaped transformer representations by the votes 
  review_rep_mean = tf.transpose(tf.expand_dims(
      tf.reduce_sum(latent_description, 0, name="review_rep_mean"),
      1))
  # final aggregated review representation
  agg_review_representation = tf.keras.layers.Dense(
      latent_size, activation=tf.keras.layers.LeakyReLU(alpha=activation_alpha),
      name="agg_review_repr")(review_rep_mean)
  agg_review_representation = tf.keras.layers.Dropout(dropout_rate)(agg_review_representation)
  #previous_bsr_input_reshaped = tf.expand_dims(tf.reduce_sum(previous_bsr_input, 0), 0)
  # concatenate all three inputs
  combined_text_meta = tf.keras.layers.concatenate(
    [agg_review_representation, previous_bsr_input],
    axis=1, name="concat_all_inputs"
  )
  # one more dense layer to account for interplay between inputs
  final_vector = tf.keras.layers.Dense(
      latent_size, activation=tf.keras.layers.LeakyReLU(alpha=activation_alpha),
      name="final_vector")(combined_text_meta)
  final_vector = tf.keras.layers.Dropout(dropout_rate)(final_vector)
  # final dense unit for the regression
  final = tf.keras.layers.Dense(
      1, activation="linear",
      name="final_answer")(final_vector)
  #final = tf.reduce_sum(tf.reduce_sum(final))
  # final model description
  model = tf.keras.models.Model(
      inputs=[text_input, previous_bsr_input],
      outputs=final, name="transformer_regression")
  return model

def create_textonly_model(transformer, latent_size=LATENT_LAYER_SIZE,
                 max_seq_length=MAX_SEQ_LENGTH,
                 activation_alpha=ACTIVATION_ALPHA,
                 dropout_rate=DROPOUT_RATE):
  """Returns model structure for regression on transformer embeddings."""
  # Input text of the form: np.array(["first review.", ...])
  text_input = tf.keras.layers.Input(shape=(None,), dtype=tf.string,
                                     name="text_input")

  # create encodings for the text
  encodings = transformer(text_input[0])
  # add additional dense layer to interpret transformer encodings and keep
  # shapes constant regardless of transformer shape
  latent_description = tf.keras.layers.Dense(
      latent_size, activation=tf.keras.layers.LeakyReLU(alpha=activation_alpha),
      name="latent_desc")(encodings)
  latent_description = tf.keras.layers.Dropout(dropout_rate)(latent_description)
  # weigh the reshaped transformer representations by the votes 
  review_rep_mean = tf.transpose(tf.expand_dims(
      tf.reduce_sum(latent_description, 0, name="review_rep_mean"),
      1))
  # final aggregated review representation
  agg_review_representation = tf.keras.layers.Dense(
      latent_size, activation=tf.keras.layers.LeakyReLU(alpha=activation_alpha),
      name="agg_review_repr")(review_rep_mean)
  agg_review_representation = tf.keras.layers.Dropout(dropout_rate)(agg_review_representation)
  # one more dense layer to account for interplay between inputs
  final_vector = tf.keras.layers.Dense(
      latent_size, activation=tf.keras.layers.LeakyReLU(alpha=activation_alpha),
      name="final_vector")(agg_review_representation)
  final_vector = tf.keras.layers.Dropout(dropout_rate)(final_vector)
  # final dense unit for the regression
  final = tf.keras.layers.Dense(
      1, activation="linear",
      name="final_answer")(final_vector)
  #final = tf.reduce_sum(tf.reduce_sum(final))
  # final model description
  model = tf.keras.models.Model(
      inputs=[text_input],
      outputs=final, name="transformer_regression")
  return model

bert_model = create_bert()
model = create_textonly_model(bert_model)
#tf.keras.utils.plot_model(model, show_shapes=True)

print("Model created...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(name="mean_squared_error"),
    metrics=[tf.keras.metrics.mean_absolute_error,
        tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))]
)

print(model.summary())

# Prediction on first row before training (random weights)
for x, y in train_generator_textonly(review_text_generator, batch_size=1, uncased=True):
  #print(x)
  print(f"Model prediction: {model.predict(x)}, Truth value: {y}")
  break

training_history = model.fit(
    train_generator_textonly(
        review_text_generator,
        batch_size=1,
        uncased=True), 
    epochs=TRAIN_EPOCHS,
    steps_per_epoch=EPOCH_STEPS)

# Prediction on first row after training
for i, (x, y) in enumerate(train_generator_textonly(review_text_generator, uncased=True)):
  print(f"{i}  Model prediction: {model.predict(x)}, Truth value: {y}")
  break

model.save('trained_model_sales_linear')

with open(f"{data}/clean/training_history.pickle", "wb") as f:
    pickle.dump(training_history, f)

with open(f"{data}/clean/test_dict.pickle", "wb") as f:
    pickle.dump(test_dict, f)

with open(f"{data}/clean/train_dict.pickle", "wb") as f:
    pickle.dump(train_dict, f)

train_df.to_pickle(f"{data}/clean/train_df.pickle")
test_df.to_pickle(f"{data}/clean/test_df.pickle")
