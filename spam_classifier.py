# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:24:03 2021

@author: Tohid(Ted) Naseri
"""


#####################################################################################
# ------------------------------- Importing Libraries------------------------------ #
#####################################################################################
from flask import Flask, render_template, request, redirect, flash
from flask_mail import Mail, Message

from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import math


import matplotlib
matplotlib.use('Agg')# For ploting in flask
import matplotlib.pyplot as plt
from io import BytesIO# For ploting in flask
import base64# For ploting in flask


import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import bs4 as bs
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, sent_tokenize
import re

import pandas as pd
import numpy as np
nltk.download('punkt')  # one time execution

# For saving and loading machine learning model
import pickle

#####################################################################################
# ------------------------------- General Functions ------------------------------- #
#####################################################################################

def plotHistogram(sentence_scores):
    """input: a dictionary of score for each sentence
    output: histogram of scores
    """
    scoreLst = list(sentence_scores.values())
    scoreLst = [x*100 for x in scoreLst]# For Visualization purposes
    #plt.hist(scoreLst)
        
    #sns.set_palette("summer")
    #sns.histplot(scoreLst, kde=True, color='red')
    
    fig, ax = plt.subplots(figsize=(5.5,4))
    n, bins, patches = plt.hist(scoreLst, bins=20, facecolor='blue', edgecolor='white', alpha=0.9)
    n = n.astype('int') # it MUST be integer
    # Good old loop. Choose colormap of your taste
    for i in range(len(patches)):
        #x = n[i]/max(n)
        x = (len(n)-1)/len(n) - i/len(n)
        #patches[i].set_facecolor(plt.cm.seismic(x))
        patches[i].set_facecolor(plt.cm.autumn(x))
        patches[i].set_facecolor(plt.cm.winter(x))
        #patches[i].set_facecolor(plt.cm.plasma(x))
        #patches[i].set_facecolor(plt.cm.Spectral(x))
        #patches[i].set_facecolor(plt.cm.hot(x))
    
    # Add title and labels with custom font sizes
    plt.title('Distribution of Sentences Score', fontsize=12)
    plt.xlabel('Score of Sentences (× 100)', fontsize=10)
        
    plt.ylabel('Count of Sentences', fontsize=10)
    
    img1 = BytesIO()
    plt.savefig(img1, format='png', bbox_inches='tight')
    plt.close()
    img1.seek(0)
    return img1
    


def plotBar(stat_df):
    """input: a dataframe including statistics of sentences score for input text and summary
    output: barplot of comparison input text vs summary
    """
    
    fig, ax = plt.subplots(figsize=(5.5,4))
    sns.barplot(x=stat_df['Variable'], y=stat_df['Value'], hue=stat_df['Type'], ax=ax);
    ax.set_title('Sentences Score Input Text vs Summary')
    ax.set_xlabel('Statistics')
    ax.set_ylabel('Score Value');
    ax.tick_params(axis='x', labelrotation=0)
    
    img2 = BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight')
    plt.close()
    img2.seek(0)
    return img2

def make_bar_plot(X, y, title='Title', xlbl='X_Label', ylbl='Y_Label', xRotation=90, annotation=False):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(X, y, ax=ax)
    #data.plot(kind='bar', legend = False, ax=ax)
    patches = ax.patches
    n = len(patches)
    if(n==2):
        patches[0].set_facecolor('cornflowerblue')
        patches[1].set_facecolor('orangered')
    else:
        for i in range(n):
            x = (n-1)/n - i/n
            patches[i].set_facecolor(plt.cm.brg(x))
            
    if(annotation == True):
        for p in ax.patches:
            x=p.get_bbox().get_points()[:,0]
            y=p.get_bbox().get_points()[1,1]
            ax.annotate('{:.1f}%'.format(y), (x.mean(), y), ha='center', va='bottom')

    if(xRotation != 0):
        ax.tick_params(axis='x', labelrotation=xRotation)
    
    #x_pos = np.arange(len(df["word"]))
    #plt.xticks(x_pos, df["word"])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlbl, fontweight="bold")
    ax.set_ylabel(ylbl, fontweight="bold")
    #plt.show()
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return img
    
    
#####################################################################################
# ------------------------------- Test_Spam Class -------------------------------------#
##################################################################################### 
class test_spam:

    def __init__(self, txtInput=""):
        """Constructor of test_spam class,
        Only one parameter is available which is the given message by user.
        The goal is evaluating the possibility of being spam or not.
        """
        self.inputMessage = txtInput

    def checkSpam(self)->dict:
        """
        input: is the given message from the user which is already saved in inputMessage.
        output: Possibility of being spam.
        """
        # Loading model from disk:
        path = "D:\\Python_DS\\Jupyter\\Spam_Classifier\\model\\model.sav"
        #loaded_model = pickle.load(open(fileName, 'rb'))
        loaded_model = pickle.load(open(path, 'rb'))
        inputLst = [""]*1;
        inputLst[0] = self.inputMessage;
        resNumeric = loaded_model.predict(inputLst)
        probLst = loaded_model.predict_proba(inputLst)[0]
        probLst *= 100
        
        if(resNumeric == 1):
            res = 'Spam Message'
        else:
            res= 'Normal Message'
        print(res)
        typeLst = ['Normal', 'Spam']
        img = make_bar_plot(X=typeLst, y=probLst, title='Statistics', xlbl='Message Type', ylbl='Probability (%)', xRotation = 0, annotation=True)
        return res, img
        
#####################################################################################
# ------------------------------- Flask Application --------------------------------#
#####################################################################################
app = Flask(__name__,static_folder="static")
    
@app.route('/test_spam', methods=['GET','POST'])
def testSpam():
    if request.method == 'POST':
        if request.form['btn'] == 'Submit':
            errorFlag=0
            try:
                txtMessage = request.form['input_message']
                if(txtMessage == ""):
                    flash('Error: No message is added. Please enter a message!')
                    errorFlag = 1
                    return render_template('test_spam.html', messageStatus = " ")
            except:
                flash('Error: Text message cannot be read. Please enter another message!')
                errorFlag = 1
                return render_template('test_spam.html', messageStatus = " ")
            if(errorFlag == 0):
                obj = test_spam(txtMessage)
                res, img = obj.checkSpam()
                img.seek(0)
                plot_url1 = base64.b64encode(img.getvalue()).decode('utf8')
                print(txtMessage)
                return render_template('test_spam.html', scroll='reportAnchor',
                                       givenMessage=txtMessage,
                                       messageStatus = res,
                                       plot_url1=plot_url1)
        
        elif request.form['btn'] == 'Reset':
            return render_template('test_spam.html', messageStatus = " ")
    else:
        return render_template('test_spam.html', messageStatus = " ")

    
@app.route('/spam_classifier', methods=['GET','POST'])
def spam_classifier():
    if request.method == 'POST':
        return render_template('spam_classifier.html')
    else:
        return render_template('spam_classifier.html')    

@app.route('/')
def home_page():
    title = 'Ted Home Page'
    return render_template('index.html', title=title)

if __name__ == '__main__':
    
    app.debug = True
    #app.run()
    app.run(debug=True, use_reloader=False)