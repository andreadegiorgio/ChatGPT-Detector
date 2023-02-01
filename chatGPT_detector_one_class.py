# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 2023

@title: One-Class ChatGPT-Detector
@author: Andrea de Giorgio
"""

#SETTINGS

#Path to the student's answers files
path_exams = r'C:/PathToDataset/MG2100_answers_students'

#Path to the ChatGPT's answers files
path_chat = r'C:/PathToDataset/MG2100_answers_chatgpt'

#Number of students' answers per exam trace
students = 36

#Number of ChatGPT chats per exam trace (3 in our dataset / 1 default)
chatGPT_chats = 3

#Number of ChatGPT's answers per exam trace and per chat (10 in our dataset)
chatGPT_attempts = 10

#Number of exam traces (6 in our dataset)
exam_answers = 6

#CODE

#Import the necessary libraries
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef

#Function to clean the imported text from files
def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('-', ' ')
    text = text.replace('/', ' ')
    text = re.sub('[^a-z0-9 ]', '', text)
    return text

#Function to import text from files
def get_content(file):
    with open(file, encoding="utf8") as f:
        return clean_text(f.read())

#Main code

#Iterate through i exam traces (6 traces in our dataset)
for i in range(1, exam_answers + 1):   
    print("Results for exam answer " + str(i) + ":")
    corpus_students = []
    corpus_chatGPT = []
    labels = []
    y_students = []
    y_chatGPT = []
    
    #Load students data
    for j in range(1, students + 1):
        corpus_students.append(get_content(path_exams + 'student' + str(j) + '_answer' + str(i) + '.txt'))
        labels.append('Answer ' + str(i) + ' Student ' + str(j))
        y_students.append(-1)
    
    #Load ChatGPT data
    for j in range(1, chatGPT_chats + 1):
        for k in range(1, chatGPT_attempts + 1):
            corpus_chatGPT.append(get_content(path_chat + 'answer' + str(i) + '_chat' + str(j) + '_attempt' + str(k) + '.txt'))
            labels.append('Answer ' + str(i) + ' Chat ' + str(j) + ' Attempt ' + str(k))
            y_chatGPT.append(1)
    
    #Tf/Idf algorithm for vectorization
    #Create a corpus with the training set
    print("Corpus generated on one-class (ChatGPT)")
    corpus = corpus_chatGPT[0:25]
    vectorizer = TfidfVectorizer()
    vect = vectorizer.fit_transform(corpus)
    X_train = vect.toarray()
    y_train = y_chatGPT[0:25]
    
    #Vectorize the test set using the previously generated corpus
    new_docs = corpus_students[0:36] + corpus_chatGPT[25:30]
    vect = vectorizer.transform(new_docs)
    X_test = vect.toarray()
    y_test = y_students[0:36] + y_chatGPT[25:30]
    
    #One class support vector machine classifier with radial basis function kernel
    print("One class support vector machine classifier")
    clf = OneClassSVM(gamma='auto')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test) #predict the class of the answers in X_test
    
    #Dispaly classification metrics based on the test set
    print(metrics.classification_report(y_test, y_pred, target_names=['Positive', 'Negative']))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    mcc = round(matthews_corrcoef(y_test, y_pred),2)
    print("MCC = " + str(mcc))
    
    #Display the confusion matrix using the test set
    cm = confusion_matrix(y_test, y_pred, labels=[1,-1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['ChatGPT','student'])
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues,ax=ax)
    disp.ax_.set_title('Answer ' + str(i))
    fig.savefig('CM_1-SVM_trace' + str(i) + '.png', dpi=200)
    
    #Display the normalized confusion matrix using the test set
    cm = confusion_matrix(y_test, y_pred, labels=[1,-1],normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['ChatGPT','student'])
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues,ax=ax)
    disp.ax_.set_title('Answer ' + str(i))
    fig.savefig('CM_norm_1-SVM_trace' + str(i) + '.png', dpi=200)
    