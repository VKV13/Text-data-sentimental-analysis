import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import nltk
import os
import glob
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from nltk import sent_tokenize

# Creating function ("article") to read each url
def article(url):

    try:
        response=requests.get(url)
        response.raise_for_status()
        soup=BeautifulSoup(response.text, 'html.parser')
        title=soup.find('title').get_text(strip=True)
        article_text=""
        main_content=soup.find_all('p')
        for paragraph in main_content:
            article_text+=paragraph.get_text(strip=True)+"\n"
        return title,article_text
    except Exception as e:
         print(f"Error fetching URL {url}:{e}")
         return None,None

# Creating function to Save the article title and content to a text file named after URL_ID.
def save_article_to_file(url_id,title,content):
    filename=f"{url_id}.txt"
    with open(filename,'w',encoding='utf-8') as file:
        file.write(title+"\n\n"+content)

# Load the Excel file
def main():
    input_file ="C:/Users/vidhy/OneDrive/Desktop/Test Assignment/Input.xlsx"
    data = pd.read_excel(input_file)
    output_dir = 'articles'
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    for index, row in data.iterrows():
        url_id=row['URL_ID']
        url=row['URL']
        title, content = article(url)
        if title and content:
            save_article_to_file(url_id, title, content)

if __name__ == "__main__":
    main()


#Creating list for negative words
negative_words=[]
with open("C:/Users/vidhy/OneDrive/Desktop/Test Assignment/negative-words.txt",'r') as file:
    for i in file:
        negative_words.append(i.strip())
#print("Negative words:",negative_words)

#Creating list for positive words
positive_words=[]
with open("C:/Users/vidhy/OneDrive/Desktop/Test Assignment/positive-words.txt",'r') as file:
    for i in file:
        positive_words.append(i.strip())
#print("Negative words:",positive_words)


#  1)           -----------------------SENTIMENTAL ANALYSIS-------------------------
#1.1	Cleaning using Stop Words Lists
#Creating set to store stop words
stop_words=set()
text_files=glob.glob(os.path.join("C:/Users/vidhy/OneDrive/Desktop/Test Assignment/Stop words",'*.txt'))
word_pattern=re.compile(r'\b[a-zA-Z]+\b')
for text_file in text_files:
    with open(text_file,'r',encoding='ISO-8859-1') as file:
        content=file.read()
        words=word_pattern.findall(content)
        for word in words:
            stop_words.add(word.lower())
#print("Stop words:",stop_words)


#  2)                  -------------CREATING DICTIONARIES FOR POSITIVE AND NEGATIVE WORDS WHICH ARE NOT IN STOP WORDS LIST------------
f_positive_words = {word for word in positive_words if word not in stop_words}
f_negative_words = {word for word in negative_words if word not in stop_words}
print("Negative words:",f_negative_words,"\nPositive words:",f_positive_words)


# 3)                ------------------------EXTRACTING DERIVED VARIABLES----------------------------------
stemmer = PorterStemmer()

# Function to clean and tokenize the text
def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if word.isalpha()]
    #tokens = [stemmer.stem(word) for w in tokens]
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

# Sentiment Analysis Scores
def calculate_sentiment_scores(text):
    tokens=tokenize(text)
    positive_score=0
    negative_score=0
    for w in tokens:
        if w in f_positive_words:
            positive_score+=1
        elif w in f_negative_words:
            negative_score+=1
    if (positive_score+negative_score)==0:
        polarity_score=0
    else:
        polarity_score=(positive_score-negative_score)/(positive_score+negative_score+0.000001)
    total_words=len(tokens)
    if total_words==0:
        subjectivity_score=0
    else:
        subjectivity_score=(positive_score+negative_score)/(total_words+0.000001)
    return positive_score,negative_score,polarity_score,subjectivity_score

# Readability Analysis (Gunning Fog Index)
def calculate_readability(text):
    sentences=sent_tokenize(text)
    words=tokenize(text)

    # Average Sentence Length
    avg_sentence_length=len(words)/len(sentences)

    # Percentage of Complex Words
    complex_words=[w for w in words if len([char for char in w if char in 'aeiou'])>2]
    percentage_complex_words=len(complex_words)/len(words)*100 if len(words)>0 else 0

    # Fog Index
    fog_index=0.4*(avg_sentence_length+percentage_complex_words)
    return avg_sentence_length,percentage_complex_words,fog_index

# Count Complex Words (More than two syllables)
def count_complex_words(text):
    words=tokenize(text)
    complex_words=[w for w in words if len([char for char in w if char in 'aeiou'])>2]
    return len(complex_words)

# Syllable Count Per Word (Handling common exceptions)
def syllable_count(word):
    # Count vowels in the word as syllables
    syllable_count=len([char for char in word if char in 'aeiou'])
    return syllable_count

def count_syllables(text):
    words=tokenize(text)
    syllables=sum(syllable_count(word) for word in words)
    return syllables

# Count Total Words
def count_words(text):
    tokens=tokenize(text)
    return len(tokens)

# Personal Pronouns Count
def count_personal_pronouns(text):
    pronouns=['i','we','my','ours','us']
    words=tokenize(text)
    pronoun_count=sum(1 for w in words if w in pronouns)
    return pronoun_count

# Average Word Length
def average_word_length(text):
    words=tokenize(text)
    total_chars=sum(len(word) for word in words)
    avg_word_len=total_chars/len(words) if len(words)>0 else 0
    return avg_word_len

# Process all files in the "article" directory
def process_articles_in_directory(directory_path, output_csv='output.csv'):
    results=[]
    for filename in os.listdir(directory_path):
        file_path=os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            try:
                with open(file_path,'r',encoding='utf-8') as file:
                    text = file.read()
                # Sentiment Scores
                positive_score,negative_score,polarity_score,subjectivity_score=calculate_sentiment_scores(text)
                # Readability Analysis
                avg_sentence_length,percentage_complex_words,fog_index=calculate_readability(text)
                # Complex Word Count
                complex_word_count=count_complex_words(text)
                # Syllable Count
                syllable_count_total=count_syllables(text)
                # Word Count
                word_count=count_words(text)
                # Personal Pronouns Count
                pronoun_count=count_personal_pronouns(text)
                # Average Word Length
                avg_word_len=average_word_length(text)
                # URL_ID and URL
                url_id=filename.split('.')[0]
                url=file_path # Full path to the file as URL
                # Store the results for the current file
                results.append([
                    url_id,url,
                    positive_score,negative_score,polarity_score,subjectivity_score,
                    avg_sentence_length,percentage_complex_words,fog_index,
                    word_count/len(sent_tokenize(text)) if len(sent_tokenize(text))>0 else 0,
                    # Average number of words per sentence
                    complex_word_count, word_count, syllable_count_total, pronoun_count,avg_word_len
                ])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    # Create a pandas DataFrame to store the results
    columns = [
        'URL_ID', 'URL',
        'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
        'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
        'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
        'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
    ]
    df = pd.DataFrame(results, columns=columns)

    # Save the results to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

directory_path="C:/Users/vidhy/PycharmProjects/Practise/articles"
process_articles_in_directory(directory_path,"C:/Users/vidhy/OneDrive/Desktop/Test Assignment/Text_analysis_output.csv")
