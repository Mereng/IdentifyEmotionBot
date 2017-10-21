from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer
import re
import numpy
import tensorflow
import tflearn
import telebot
import config


class NN:

    def __init__(self):
        self._stem_cache = {}
        self._validator_regex = re.compile(r'[^А-яЁё]')
        self._stemmer = RussianStemmer()
        self.vocabulary = self._load_vocabulary()
        self.model = self._load_model()

    def _get_stem(self, token):
        stem = self._stem_cache.get(token, None)

        if stem:
            return stem

        token = self._validator_regex.sub('', token).lower()
        stem = self._stemmer.stem(token)
        self._stem_cache[token] = stem
        return stem

    def _load_vocabulary(self):
        with open('data/vocabulary.txt') as f:
            vocabulary_arr = f.read().split('\n')
        return {vocabulary_arr[i] : i for i in range(len(vocabulary_arr))}

    def message_to_vector(self, message):
        vector = numpy.zeros(len(self.vocabulary), dtype=numpy.byte)
        tokenizer = TweetTokenizer()
        for token in tokenizer.tokenize(message):
            stem = self._get_stem(token)
            idx = self.vocabulary.get(stem, None)
            if idx is not None:
                vector[idx] = 1
        return vector

    def _build_model(self):
        tensorflow.reset_default_graph()
        net = tflearn.input_data([None, len(self.vocabulary)])
        net = tflearn.fully_connected(net, 125, activation='RelU')
        net = tflearn.fully_connected(net, 25, activation='RelU')
        net = tflearn.fully_connected(net, 2, activation='softmax')

        return tflearn.DNN(net)

    def _load_model(self):
        model = self._build_model()
        model.load('data/model/model3')
        return model

    def take_answer(self, message):
        vector = [self.message_to_vector(message)]
        return self.model.predict(vector)[0][1] >= 0.5


nn = NN()
bot = telebot.TeleBot(config.token)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет! Пиши мне, а я скажу позитивный ли это текст или негативный =)')


@bot.message_handler(content_types=['text'])
def answer_emotion(message):
    if nn.take_answer(message.text):
        answer = 'Позитив'
    else:
        answer = 'Негатив'
    bot.send_message(message.chat.id, answer)


if __name__ == '__main__':
    bot.polling(none_stop=True)