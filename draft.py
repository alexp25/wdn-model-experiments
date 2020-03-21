model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(time_step, data_dim)))
model.add(LSTM(32, return_sequences=True))
# model.add(Dense(output_dim, activation='linear'))
model.add(Dense(output_dim, activation='sigmoid'))

# There are three kinds of classification tasks:

# Binary classification: two exclusive classes
# Multi-class classification: more than two exclusive classes
# Multi-label classification: just non-exclusive classes
# Here, we can say
# In the case of (1), you need to use binary cross entropy.
# In the case of (2), you need to use categorical cross entropy.
# In the case of (3), you need to use binary cross entropy.

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
