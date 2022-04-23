from collections import Counter

from pyparsing import WordStart
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import requests
from io import StringIO
import math


class PreprocessData:
    def __init__(self,path,lang='english'): 
        self.dataset=pd.read_csv(path, sep=",")
        self.stopwords=stopwords.words(lang)
        self.preprocess()
    
    @staticmethod
    def remove_punctuation(text):
        '''a function for removing punctuation'''
        # replacing the punctuations with no space, 
        # which in effect deletes the punctuation marks 
        translator = str.maketrans('', '', string.punctuation)
        # return the text stripped of punctuation marks
        return text.translate(translator)

    def replace_sentiment(self):
    #Convert the sentiment to numerical value......1 for positive and 0 for negative
        self.dataset = self.dataset.replace({'sentiment': {'positive': 1, 'negative': 0}})

    #A function to remove the stopwords
    def remove_stopwords(self,text):
        text = [word.lower() for word in text.split() if word.lower() not in self.stopwords]
        # joining the list of words with space separator
        return " ".join(text)

    def preprocess(self):
        self.dataset['review'] = self.dataset['review'].apply(self.remove_punctuation)
        self.dataset['review'] = self.dataset['review'].apply(self.remove_stopwords)
        self.replace_sentiment()





class WordRepresentation:
    def __init__(self,dataset):
        self.dataset=dataset
        self.ponderations={}
        self.vectors={}
        self.frequencies={}
        self.length=0 #The number of words on the dataset
        self.tf_idf()
        self.one_hot()
        self.frequency()
    
    #One hot
    def one_hot(self):
        if self.ponderations or self.frequencies:
            #The case if we have the ponderations or the frequencies , we don't need to search the words again
            #This condition will reduce the time complexity a lot   
            words_dict={}
            if self.ponderations:
                words_dict=self.ponderations
            else:
                words_dict=self.frequencies
            for word in words_dict:
                self.vectors[word]=np.zeros(len(self.dataset))
        else:
            for i in range(len(self.dataset)):
                document=self.dataset.iloc[i,0]
                word_list=document.split(" ")
                for word in word_list:
                    if word not in self.vectors:
                        self.vectors[word]=np.zeros(len(self.dataset))
        #Now we have all the words initialized as vector zero of size the length of the dataset
        for word in self.vectors:
            for i in range(len(self.dataset)):
                document=self.dataset.iloc[i,0]
                if word in document:
                    self.vectors[word][i]=1




    #Frequency encoding
    def frequency(self):
        if self.ponderations or self.vectors:
            #The case if we have the ponderations or the frequences =, we don't need to search the words again
            #This condition will reduce the time complexity a lot
            
            words_dict={}
            if self.ponderations:
                words_dict=self.ponderations
            else:
                words_dict=self.vectors
            for word in words_dict:
                self.frequencies[word]=np.zeros(len(self.dataset))
        else:
            for i in range(len(self.dataset)):
                document=self.dataset.iloc[i,0]
                word_list=document.split(" ")
                for word in word_list:
                    if word not in self.frequencies:
                        self.frequencies[word]=np.zeros(len(self.dataset))
        #Now we have all the words initialized as zero scalar
        self.compute_length()
        for word in self.frequencies:
            for i in range(len(self.dataset)):
                document=self.dataset.iloc[i,0]
                if word in document:
                    self.frequencies[word][i]=document.count(word)/self.length
    
    
    #This function takes a word and count the number of occurences of the word in the document
    # def count_word(self,word,document):
    #     sum=0
    #     for i in range(len(self.dataset)):
    #         document=self.dataset.iloc[i,0]
    #         sum+=document.count(word)
    #     return sum
    #This function compute the length 
    def compute_length(self):
        for i in range(len(self.dataset)):
            document=self.dataset.iloc[i,0]
            word_list=document.split(" ")
            self.length+=len(word_list)
    

    



    #TF_IDF
    def term_frequency(self,word,document):
        #Frequence of the word over number of word of document
        word_list=document.split(" ")
        dict=Counter(word_list) #Give a dictionnary which the keys are the words and the values are the number of occurences in word_list
        return dict[word]/len(word_list)
    
    def inverse_document_frequency(self,word,document):
        N=len(self.dataset)
        #Contains return a list of True or False
        doc_t=len(self.dataset[self.dataset['review'].str.contains(word)])
        return math.log(N/doc_t)

    def tf_idf(self):
        if self.vectors or self.frequencies:
            #The case if we have the vectors or the frequencies , we don't need to search the words again
            #This condition will reduce the time complexity a lot   
            words_dict={}
            if self.vectors:
                words_dict=self.vectors
            else:
                words_dict=self.frequencies
            for word in words_dict:
                self.vectors[word]=np.zeros(len(self.dataset))
        else:
            for i in range(len(self.dataset)):
                document=self.dataset.iloc[i,0]
                word_list=document.split(" ")
                for word in word_list:
                    if word not in self.ponderations:
                        self.ponderations[word]=np.zeros(len(self.dataset))

        for word in self.ponderations:
            for i in range(len(self.dataset)):
                document=self.dataset.iloc[i,0]
                if word in document:
                    self.ponderations[word][i]=self.term_frequency(word,document)*self.inverse_document_frequency(word,document)






path="/Users/aba/Desktop/BNB/IMDB_Dataset.csv"
documents=PreprocessData(path)
words=WordRepresentation(documents.dataset.iloc[:5,:])
print(words.frequencies)
#vectors=words.vectors