# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 2023

@title: Cosine similarity analysis
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

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#Cosine similarity function
#Note that we use the fuction from sklearn instead of this one
def cosine_similarity_calc(vec_1,vec_2):
	sim = np.dot(vec_1,vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2))
	return sim

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
    
#Function to create the cosine similarity heatmap
def create_heatmap(similarity, cmap = "YlGnBu"):
    df = pd.DataFrame(similarity)
    df.columns = labels
    df.index = labels
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(df, cmap=cmap)

#Main code

#Iterate through i exam traces (6 traces in our dataset)
for i in range(1, exam_answers + 1):
    content = []
    labels = []

    #Load students data
    for j in range(1, students + 1):
        content.append(get_content(path_exams + 'student' + str(j) + '_answer' + str(i) + '.txt'))
        labels.append('Answer ' + str(i) + ' Student ' + str(j))

    #Load ChatGPT data
    for j in range(1, chatGPT_chats + 1):
        for k in range(1, chatGPT_attempts + 1):
            content.append(get_content(path_chat + 'answer' + str(i) + '_chat' + str(j) + '_attempt' + str(k) + '.txt'))
            labels.append('Answer ' + str(i) + ' Chat ' + str(j) + ' Attempt ' + str(k))
    
    #Tf/Idf algorithm for vectorization
    #Create a corpus with the entire dataset
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(content)
    arr = X.toarray()
    
    #Calculate the cosine similarity between all the entries for exam trace i
    results = cosine_similarity(arr)
    
    #Display the cosine similarity heatmap for exam trace i
    create_heatmap(results)
    
    #Calculate the cosine similarity for the main clusters in the heatmap
    results_ss = results[0:36,0:36] #student-student
    results_cc = results[36:66,36:66] #ChatGPT-ChatGPT
    results_sc = results[0:36,36:66] #student-ChatGPT (=ChatGPT-student)
    results_cs = results[36:66,0:36] #ChatGPT-student (=student-ChatGPT)
    ssavg = round(results_ss.mean(),2) #student-student mean
    ccavg = round(results_cc.mean(),2) #ChatGPT-ChatGPT mean
    scavg = round(results_sc.mean(),2) #student-ChatGPT (=ChatGPT-student) mean
    csavg = round(results_cs.mean(),2) #ChatGPT-student (=student-ChatGPT) mean
    ssmin = round(results_ss.min(),2) #student-student min
    ccmin = round(results_cc.min(),2) #ChatGPT-ChatGPT min
    scmin = round(results_sc.min(),2) #student-ChatGPT (=ChatGPT-student) min
    csmin = round(results_cs.min(),2) #ChatGPT-student (=student-ChatGPT) min
    for h in range(0, len(results_ss)):
        results_ss[h,h] = 0 #Remove the 1s from the diagonal to calculate max
    for h in range(0, len(results_cc)):
        results_cc[h,h] = 0 #Remove the 1s from the diagonal to calculate max     
    ssmax = round(results_ss.max(),2) #student-student max
    ccmax = round(results_cc.max(),2) #ChatGPT-ChatGPT max
    scmax = round(results_sc.max(),2) #student-ChatGPT (=ChatGPT-student) max
    csmax = round(results_cs.max(),2) #ChatGPT-student (=student-ChatGPT) max
    
    #Display the cosine similarity scores for the main clusters in the heatmap
    print('For answer ' + str(i) + ':')
    
    print('student-student cosine similarity avg is: ' + str(ssavg))
    print('chatGPT-chatGPT cosine similarity avg is: ' + str(ccavg))
    print('student-chatGPT cosine similarity avg is: ' + str(scavg))
    print('chatGPT-student cosine similarity avg is: ' + str(csavg))
    
    print('student-student cosine similarity min is: ' + str(ssmin))
    print('chatGPT-chatGPT cosine similarity min is: ' + str(ccmin))
    print('student-chatGPT cosine similarity min is: ' + str(scmin))
    print('chatGPT-student cosine similarity min is: ' + str(csmin))
        
    print('student-student cosine similarity max is: ' + str(ssmax))
    print('chatGPT-chatGPT cosine similarity max is: ' + str(ccmax))
    print('student-chatGPT cosine similarity max is: ' + str(scmax))
    print('chatGPT-student cosine similarity max is: ' + str(csmax))