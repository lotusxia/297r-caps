import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.keras import TqdmCallback

import tensorflow_text as text
import tensorflow_hub as hub

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

import tensorflow_addons as tfa

# input folders
data = "./data"

# Loading the review texts
review_df = pd.read_pickle(f'{data}/prod_level_bsr_rev.pickle')
review_df = review_df[['asin', 'review_text_3_mo', 'label_after_1_yr_period_12_mo_min_bsr']]
print(len(review_df))
print(review_df.head())

review_df = review_df.dropna()

with open(f"{data}/product_sample_long_term.pickle", "rb") as f:
    product_sample = pickle.load(f)

train_asins = set(product_sample['train'])
test_asins = set(product_sample['test'])

print("DF size before filtering:", len(review_df))
train_df = review_df[review_df['asin'].isin(train_asins)]
test_df = review_df[review_df['asin'].isin(test_asins)]
print("train size after filtering: ", len(train_df))
print("test size after filtering: ", len(test_df))

print("Number of ones:", sum(train_df['label_after_1_yr_period_12_mo_min_bsr']) + sum(test_df['label_after_1_yr_period_12_mo_min_bsr']))

# Freeing up RAM
del review_df

train_data = train_df.to_dict('records')
test_data = test_df.to_dict('records')

def data_generator(data=train_data, uncased=True, validation=False, get_asins=False):
  while True:
    for data_dict in data:
      asin = data_dict['asin']
      label = data_dict['label_after_1_yr_period_12_mo_min_bsr']
      reviews = data_dict['review_text_3_mo']
      if len(reviews)==0: continue
      if uncased: reviews = [review_text.lower() for review_text in reviews]
      reviews = np.expand_dims(np.array(reviews), 0)
      label = np.expand_dims(np.array([label]), 0)
      if not get_asins: yield [reviews], label
      else: yield [reviews], label, asin

# Common Parameters
ACTIVATION_ALPHA = 0.04
EPOCH_STEPS = len(train_data)
DROPOUT_RATE = 0.1
TRAIN_EPOCHS = 10
MAX_SEQ_LENGTH = 128
LATENT_LAYER_SIZE = 10

for TRAIN_EPOCHS in [2, 5, 10]:
	for LATENT_LAYER_SIZE in [10, 40, 100]:

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
		    # create encodings for the text
		    encodings = transformer(text_input[0])
		    # add additional dense layer to interpret transformer encodings and keep
		    # shapes constant regardless of transformer shape
		    latent_description = tf.keras.layers.Dense(
		        latent_size, activation=tf.keras.layers.LeakyReLU(alpha=activation_alpha),
		        name="latent_desc1")(encodings)
		    latent_description = tf.keras.layers.Dropout(dropout_rate)(latent_description)
		    latent_description = tf.keras.layers.Dense(
		        latent_size, activation=tf.keras.layers.LeakyReLU(alpha=activation_alpha),
		        name="latent_desc2")(latent_description)
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
		        name="final_vector1")(agg_review_representation)
		    final_vector = tf.keras.layers.Dropout(dropout_rate)(final_vector)
		    final_vector = tf.keras.layers.Dense(
		        latent_size, activation=tf.keras.layers.LeakyReLU(alpha=activation_alpha),
		        name="final_vector2")(final_vector)
		    final_vector = tf.keras.layers.Dropout(dropout_rate)(final_vector)
		    # final dense unit for the regression
		    final = tf.keras.layers.Dense(
		        1, activation="sigmoid",
		        name="final_answer")(final_vector)
		    #final = tf.reduce_sum(tf.reduce_sum(final))
		    # final model description
		    model = tf.keras.models.Model(
		        inputs=[text_input],
		        outputs=final, name="transformer_regression")
		    return model

		bert_model = create_bert()
		model = create_model(bert_model)

		print("Model created...")

		model.compile(
		    optimizer=tf.keras.optimizers.Adam(),
		    loss=tf.keras.losses.BinaryCrossentropy(),
		    metrics=[tf.keras.metrics.AUC(name='auc'),
		            tfa.metrics.F1Score(name="f1", num_classes=1),
		            tf.keras.metrics.Precision(name="precision"),
		            tf.keras.metrics.Recall(name="recall"),
		            tf.keras.metrics.Accuracy(name='accuracy')])

		print(model.summary())

		training_history = model.fit(
		    data_generator(data=train_data, uncased=True),
		    validation_data=data_generator(data=test_data, uncased=True),
		    epochs=TRAIN_EPOCHS,
		    steps_per_epoch=EPOCH_STEPS,
		    validation_steps=len(test_data),
		    verbose=0, callbacks=[TqdmCallback(verbose=1)])


		model.save('long_term_final2')

		results = model.evaluate(
		    data_generator(data=test_data, uncased=True),
		    steps=len(test_data)
		)

		model_row = ""
		model_row += "layer_depth, " + str(1) + ", "
		model_row += "epochs, " + str(TRAIN_EPOCHS) + ", "
		model_row += "max_seq_length, " + str(MAX_SEQ_LENGTH) + ", "
		model_row += "latent_layer_size, " + str(LATENT_LAYER_SIZE) + ", "
		model_row += "loss, " + str(results[0]) + ", "
		model_row += "f1, " + str(results[1][0]) + ", "
		model_row += "precision, " + str(results[2]) + ", "
		model_row += "recall, " + str(results[3]) + ", "
		model_row += "accuracy, " + str(results[4]) + ", "
		model_row += "auc, " + str(results[5]) + "\n"

		print(model_row)

		print("DONE.")

y_preds = []
for i, (reviews, label, asin) in enumerate(data_generator(get_asins=True)):
    if i>=len(train_data): break
    try:
        y_pred = model.predict(reviews)
        y_preds.append({"prediction":y_pred[0][0], "asin":asin, "y_true":label[0][0]})
    except: pass

for i, (reviews, label, asin) in enumerate(data_generator(data=test_data, get_asins=True)):
    if i>=len(test_data): break
    try:
        y_pred = model.predict(reviews)
        y_preds.append({"prediction":y_pred[0][0], "asin":asin, "y_true":label[0][0]})
    except: pass
print()

df = pd.DataFrame.from_dict(y_preds)
print("Number of ones:", sum(df['y_true']))

df.to_pickle("bert_res_df.pickle")

with open("saved_preds.pickle", "wb") as f:
    pickle.dump(y_preds, f)

