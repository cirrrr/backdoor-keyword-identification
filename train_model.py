import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def train_model(max_words, embedding_dim, maxlen, class_num, train, test, epochs=25, embedding_matrix=None):
    train_data, train_labels = train
    test_data, test_labels = test
    model = Sequential()
    model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(layers.Bidirectional(layers.LSTM(128)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(class_num, activation='softmax'))
   
    #determine if pre-trained word embeddings are used 
    #based on parameter "embedding_matrix"
    if np.asarray((embedding_matrix != None)).all():
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.fit(train_data, train_labels,
                    epochs=epochs,
                    batch_size=128,
                    validation_data=(test_data, test_labels),
                    verbose=0)
    return model
