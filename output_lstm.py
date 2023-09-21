import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

text="FREE RINGTONE text FIRST to 87131 for a poly or text GET to 87131 for a true tone! Help? 0845 2814032 16 after 1st free, tones are 3x�150pw to e�nd txt stop"
loaded_model = tf.keras.models.load_model('lstm_model.h5')

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequence = tokenizer.texts_to_sequences([text])
max_sequence_length = loaded_model.input_shape[1]
padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

predicted_class = loaded_model.predict(padded_sequence)
print("predicted_class :",predicted_class)
import numpy as np
ans= np.round(predicted_class)

if ans==1:
    stmt="not spam"
else:
    stmt="spam"
print(f"Predicted Class Probabilities: {stmt}")


import pyttsx3 as tts
eng=tts.init()
text="This sample text is classified as "
text=text+stmt
voices=eng.getProperty('voices')
eng.setProperty('voice',voices[1].id)
#female voice zira
for voice in eng.getProperty('voices'):
    print(voice)
eng.setProperty('rate',150)
eng.say(text)
eng.runAndWait()



