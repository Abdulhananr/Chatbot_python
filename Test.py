import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
# load the saved model file
model = load_model('chatbot.h5')
intents = json.loads(open("intents.json", encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):

    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):

    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
               
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
   
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    error = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>error]
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
    # function to get the response from the model

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


questions = [
    "Tell me about the curriculum of IsI.",
    "Introduce the curriculum of ISI.",
    "What are the degrees offered at ISI?",
    "What are the diplomas offered at ISI?",
    "Introduce ISI's code of conduct.",
    "Give me an overview of the kindergarten at ISI.",
    "What is the role of parents in ISI?",
    "What are the learning methods used at ISI?",
    "Summarize the responsibilities of parents at ISI.",
    "What is the school's schedule?",
    "Do I have to attend every day?",
    "do I have to wear uniform?"
    "What items are banned at ISI?",
    "Can I bring a tablet to school?",
    "Can I bring a bicycle to school?",
    "Do I need books? Where do I get them?",
    "Do grade 1 students have semester exams?",
    "How many exams do grade 7 students have?",
    "How many stars do I get if my average is 93?",
    "What does SP mean? When do we have them? Put them in the list."
]
import time
i=1
for item in questions:
    print(f"Question {i} -{item}\n")
    print(f"ChatBot: {chatbot_response(item)}")
    i=i+1
    time.sleep(1)