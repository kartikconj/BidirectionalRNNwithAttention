import numpy as np
import pandas as pd
import keras

data=pd.read_csv('emotion.data')
data.emotions.value_counts().plot.bar()


input_sentences= [text.split() for text in data["text"].values.tolist()]
labels=data["emotions"].values.tolist()

word2id = dict()
label2id = dict()
a=set()

max_words = 0

for sentence in input_sentences:
    for word in sentence:
        if word not in word2id:
            word2id[word]=len(word2id)
    if len(sentence)>max_words:
        max_words=len(sentence)


label2id={l:i for i, l in enumerate(set(labels))}
id2label={v:k for k,v in label2id.items()}

X=[[word2id[word] for word in sentence] for sentence in input_sentences]
Y=[label2id[label] for label in labels]

from keras.preprocessing.sequence import pad_sequences

X = pad_sequences(X, max_words)
Y = keras.utils.to_categorical(Y, num_classes=len(label2id), dtype='float32')

print("Shape of X: {}".format(X.shape))
print("Shape of Y: {}".format(Y.shape))

embedding_dim = 100

#input
sequence_input = keras.Input(shape=(max_words,), dtype='int32')
# Word embedding
embedded_inputs =keras.layers.Embedding(len(word2id) + 1,embedding_dim, input_length=max_words)(sequence_input)
# Apply dropout to prevent overfitting
embedded_inputs = keras.layers.Dropout(0.2)(embedded_inputs)
# Apply Bidirectional LSTM over embedded inputs
lstm_outs = keras.layers.wrappers.Bidirectional(
    keras.layers.LSTM(embedding_dim, return_sequences=True))(embedded_inputs)
# Apply dropout to LSTM outputs to prevent overfitting
lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)
# Attention Mechanism - Generate attention vectors
input_dim = int(lstm_outs.shape[2])
permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])
# Last layer: fully connected with softmax activation
fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
output = keras.layers.Dense(len(label2id), activation='softmax')(fc)
model = keras.Model(inputs=[sequence_input], outputs=output)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')
model.summary()
model.fit(X, Y, epochs=2, batch_size=64, validation_split=0.1, shuffle=True)
model_with_attentions = keras.Model(inputs=model.input,
                                    outputs=[model.output,
                                             model.get_layer('attention_vec').output])
import random
import math
import random
# Select random samples to illustrate
sample_text = random.choice(data["text"].values.tolist())
tokenized_sample = sample_text.split(" ")
encoded_samples = [[word2id[word] for word in tokenized_sample]]
encoded_samples = keras.preprocessing.sequence.pad_sequences(encoded_samples, maxlen=max_words)
label_probs, attentions = model_with_attentions.predict(encoded_samples)
label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(),label_probs[0])}