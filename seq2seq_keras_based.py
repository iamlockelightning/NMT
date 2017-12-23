#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re, time
import numpy as np
import nltk
from keras.models import Model
from keras.layers import Input, LSTM, Dense

to_lower_ru = {}
to_upper_ru = {}
upper_letter = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
lower_letter = ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
for i in range(len(upper_letter)):
    to_lower_ru[upper_letter[i]] = lower_letter[i]
    to_upper_ru[lower_letter[i]] = upper_letter[i]

def lower_ru(s):
    t = []
    pos = 0
    while pos < len(s):
        if s[pos] == ' ':
            t.append(' ')
            pos += 1
        else:
            if s[pos : pos+2] in upper_letter:
                t.append(to_lower_ru[s[pos : pos+2]])
            else:
                t.append(s[pos : pos+2])
            pos += 2
    return ''.join(t)

def read_corpus(filename):
    en_parallel_sentences = []
    ru_parallel_sentences = []
    with open(filename, 'r') as fr:
        for line in fr:
            line = line.strip()
            if line[0] == '#':
                tmp_1 = line.split(' - ')[0][1:].lower().strip().split()
                tmp_1.insert(0, "<s>")
                tmp_1.append("<e>")
                tmp_2 = lower_ru(line.split(' - ')[1]).strip().split()
                tmp_2.insert(0, "<s>")
                tmp_2.append("<e>")
                en_parallel_sentences.append(tmp_1)
                ru_parallel_sentences.append(tmp_2)
            else:
                [rus, eng] = line.split('\t')
                rus_sens = re.split('\.|:|!|\?', rus)
                eng_sens = re.split('\.|:|!|\?', eng)
                if len(rus_sens) == len(eng_sens):
                    for i in range(len(rus_sens)):
                        if eng_sens[i]=='':
                            continue
                        tmp_1 = eng_sens[i].lower().replace(',', ' ,').strip().split()
                        # tmp_1.insert(0, "<s>")
                        # tmp_1.append("<e>")
                        tmp_2 = lower_ru(rus_sens[i]).replace(',', ' ,').strip().split()
                        tmp_2.insert(0, "<s>")
                        tmp_2.append("<e>")
                        en_parallel_sentences.append(tmp_1)
                        ru_parallel_sentences.append(tmp_2)
                # print len(rus_sens), len(eng_sens)
                # if len(rus_sens) != len(eng_sens):
                #   for i in range(min(len(rus_sens), len(eng_sens))):
                #       print rus_sens[i]
                #       print eng_sens[i]
                #       print
    print "len(en_parallel_sentences):", len(en_parallel_sentences), "\tlen(ru_parallel_sentences):", len(ru_parallel_sentences)
    en_vocab = ['<unk>']
    ru_vocab = ['<unk>']
    for i in range(len(en_parallel_sentences)):
        for w in en_parallel_sentences[i]:
            if w not in en_vocab:
                en_vocab.append(w)
        for w in ru_parallel_sentences[i]:
            if w not in ru_vocab:
                ru_vocab.append(w)
    en_vocab = sorted(en_vocab)
    en_word2id = dict([(word, i) for i, word in enumerate(en_vocab)])
    en_id2word = dict([(str(i), word) for i, word in enumerate(en_vocab)])
    ru_vocab = sorted(ru_vocab)
    ru_word2id = dict([(word, i) for i, word in enumerate(ru_vocab)])
    ru_id2word = dict([(str(i), word) for i, word in enumerate(ru_vocab)])
    print "len(en_vocab):", len(en_vocab), "\tlen(ru_vocab):", len(ru_vocab)
    return en_parallel_sentences, ru_parallel_sentences, en_vocab, ru_vocab, en_word2id, en_id2word, ru_word2id, ru_id2word

def read_sentence_pairs(filename):
    en_parallel_sentences = []
    ru_parallel_sentences = []
    cnt = 0
    with open(filename, 'r') as fr:
        for line in fr:
            line = line.strip()
            [eng, rus] = line.split('\t')
            eng_sen = eng.lower().replace(';', ' ;').replace(',', ' ,').replace('.', ' .').replace(':', ' :').replace('!', ' !').replace('?', ' ?').split()
            rus_sen = lower_ru(rus).replace(';', ' ;').replace(',', ' ,').replace('.', ' .').replace(':', ' :').replace('!', ' !').replace('?', ' ?').split()
            # eng_sen.insert(0, "<s>")
            # eng_sen.append("<e>")
            rus_sen.insert(0, "<s>")
            rus_sen.append("<e>")
            en_parallel_sentences.append(eng_sen)
            ru_parallel_sentences.append(rus_sen)
            cnt += 1
            # if cnt == 4000:
                # break
    en_vocab = ['<unk>']
    ru_vocab = ['<unk>']
    for i in range(len(en_parallel_sentences)):
        if i%10000 == 0:
            print "__calc:", i
        for w in en_parallel_sentences[i]:
            if w not in en_vocab:
                en_vocab.append(w)
        for w in ru_parallel_sentences[i]:
            if w not in ru_vocab:
                ru_vocab.append(w)
    en_vocab = sorted(en_vocab)
    en_word2id = dict([(word, i) for i, word in enumerate(en_vocab)])
    en_id2word = dict([(str(i), word) for i, word in enumerate(en_vocab)])
    ru_vocab = sorted(ru_vocab)
    ru_word2id = dict([(word, i) for i, word in enumerate(ru_vocab)])
    ru_id2word = dict([(str(i), word) for i, word in enumerate(ru_vocab)])
    print "len(en_vocab):", len(en_vocab), "\tlen(ru_vocab):", len(ru_vocab)
    return en_parallel_sentences, ru_parallel_sentences, en_vocab, ru_vocab, en_word2id, en_id2word, ru_word2id, ru_id2word

class NMT:
    def __init__(self, src_vocab, tgt_vocab, src_word2id, tgt_word2id, src_id2word, tgt_id2word):   
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_word2id = src_word2id
        self.tgt_word2id = tgt_word2id
        self.src_id2word = src_id2word
        self.tgt_id2word = tgt_id2word
        self.hidden_dim = 256
        self.batch_size = 128
        self.epochs = 15
        self.validation_split = 0.2
        self.build_model()

    def build_model(self):
        encoder_inputs = Input(shape=(None, len(self.src_vocab)))
        self.encoder_lstm = LSTM(self.hidden_dim, return_state=True)
        encoder_outputs, en_state_h, en_state_c = self.encoder_lstm(encoder_inputs)
        encoder_states = [en_state_h, en_state_c]
        decoder_inputs = Input(shape=(None, len(self.tgt_vocab)))
        self.decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, state_h, state_c = self.decoder_lstm(decoder_inputs, initial_state=encoder_states)
        self.decoder_dense = Dense(len(self.tgt_vocab), activation='softmax')
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(self.hidden_dim,))
        decoder_state_input_c = Input(shape=(self.hidden_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, de_state_h, de_state_c = self.decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [de_state_h, de_state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def train(self, src_parallel_sentences, tgt_parallel_sentences):
        self.src_sen_max_len = max([len(t) for t in src_parallel_sentences])
        self.tgt_sen_max_len = max([len(t) for t in tgt_parallel_sentences])
        encoder_input_data = np.zeros((len(src_parallel_sentences), self.src_sen_max_len, len(self.src_vocab)), dtype='float32')
        decoder_input_data = np.zeros((len(src_parallel_sentences), self.tgt_sen_max_len, len(self.tgt_vocab)), dtype='float32')
        decoder_target_data = np.zeros((len(src_parallel_sentences), self.tgt_sen_max_len, len(self.tgt_vocab)), dtype='float32')

        for i in range(len(src_parallel_sentences)):
            for j in range(len(src_parallel_sentences[i])):
                encoder_input_data[i, j, self.src_word2id[src_parallel_sentences[i][j]]] = 1.0
            for j in range(len(tgt_parallel_sentences[i])):
                decoder_input_data[i, j, self.tgt_word2id[tgt_parallel_sentences[i][j]]] = 1.0
                if j > 0:
                    decoder_target_data[i, j-1, self.tgt_word2id[tgt_parallel_sentences[i][j]]] = 1.0

        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split)
        filename = "en2ru."+time.strftime("%Y%m%d_%H%M%S", time.localtime())+".h5"
        self.model.save(filename)
        print "Model saved as:", filename

    def evaluate(self, src_parallel_sentences, tgt_parallel_sentences):
        reference = []
        hypothesis = []
        for i in range(len(src_parallel_sentences)):
            reference.append([tgt_parallel_sentences[i]])
            hypothesis.append(self.reference(src_parallel_sentences[i]))
        BLEU = nltk.translate.bleu_score.corpus_bleu(reference, hypothesis, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        print "BLEU score:", BLEU

    def reference(self, input_seq):
        encoder_input_data = np.zeros((len(input_seq), self.src_sen_max_len, len(self.src_vocab)), dtype='float32')
        for i, w in enumerate(input_seq):
            if w in self.src_word2id:
                encoder_input_data[0, i, self.src_word2id[w]] = 1.0
            else:
                encoder_input_data[0, i, self.src_word2id['<unk>']] = 1.0
        states_value = self.encoder_model.predict(encoder_input_data)
        target_seq = np.zeros((1, 1, len(self.tgt_vocab)))
        target_seq[0, 0, self.tgt_word2id["<s>"]] = 1.0
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.tgt_id2word[str(sampled_token_index)]
            decoded_sentence.append(sampled_word)
            if (sampled_word == "<e>" or len(decoded_sentence) > self.tgt_sen_max_len):
                stop_condition = True
            target_seq = np.zeros((1, 1, len(self.tgt_vocab)))
            target_seq[0, 0, sampled_token_index] = 1.0
            states_value = [h, c]
        return ' '.join(decoded_sentence)


if __name__ == "__main__":
    # en_parallel_sentences, ru_parallel_sentences, en_vocab, ru_vocab, en_word2id, en_id2word, ru_word2id, ru_id2word = read_corpus("corpus.txt")
    en_parallel_sentences, ru_parallel_sentences, en_vocab, ru_vocab, en_word2id, en_id2word, ru_word2id, ru_id2word = read_sentence_pairs("nmt/rus-eng/rus.txt")
    nmt = NMT(en_vocab, ru_vocab, en_word2id, ru_word2id, en_id2word, ru_id2word)
    sentences = zip(en_parallel_sentences, ru_parallel_sentences)
    np.random.shuffle(sentences)
    en_parallel_sentences, ru_parallel_sentences = zip(*sentences)
    nmt.train(list(en_parallel_sentences), list(ru_parallel_sentences))
    nmt.evaluate(en_parallel_sentences[1:3000], ru_parallel_sentences[1:3000])
    # print nmt.reference("I saw Tom .".lower().split())

