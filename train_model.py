import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def train_model(max_words, embedding_dim, maxlen, class_num, train, test, embedding_matrix=None):
    train_data, train_labels = train
    model = Sequential()
    model.add(layers.Embedding(max_words, embedding_dim, mask_zero=False, input_length=maxlen))
    model.add(layers.Bidirectional(layers.LSTM(128)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(class_num, activation='softmax'))
   
    #determine if pre-trained word embeddings are used 
    #based on parameter "embedding_matrix"
    if np.asarray((embedding_matrix != None)).all():
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.fit(train_data, train_labels,
                    epochs=25,
                    batch_size=64,
                    validation_data=test)
    return model
