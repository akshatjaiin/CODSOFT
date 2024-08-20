import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

df_train = pd.read_csv("Genre Classification Dataset/train_data.txt",sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
x_test = pd.read_csv("Genre Classification Dataset/test_data.txt",sep=':::', names=['ID', 'TITLE', 'DESCRIPTION'])
df_test_sol= pd.read_csv("Genre Classification Dataset/test_data_solution.txt",sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])

df_train.head(3)

x_test.head(3)

df_test_sol

df_train.info()
df_test_sol.info()

import matplotlib.pyplot as plt

genre_counts = df_train['GENRE'].value_counts()

plt.figure(figsize=(20,8))

plt.bar(genre_counts.index, genre_counts.values,color=['red', 'green', 'blue', 'orange', 'purple'])
plt.title('Number of Movies per Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)  # Rotate genre labels for better readability
plt.tight_layout()
plt.grid()
plt.show()

most_watched_genre = genre_counts.idxmax()

print("The most watched genre is:", most_watched_genre)

df_train=df_train.drop(columns=['ID'],axis=1)
x_test=x_test.drop(columns=['ID'],axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['GENRE'] = le.fit_transform(df_train['GENRE'])

df_test_sol['GENRE'] = le.fit_transform(df_test_sol['GENRE'])
df_train['combined_text'] = df_train['TITLE'] + ' ' + df_train['DESCRIPTION']
x_test['combined_text'] = x_test['TITLE'] + ' ' + x_test['DESCRIPTION']
X_train=df_train.drop(['GENRE','DESCRIPTION','TITLE'],axis=1)

X_test=x_test.drop(['DESCRIPTION','TITLE'],axis=1)
y_train=df_train['GENRE']
y_test=df_test_sol['GENRE']
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer on X_train
tfidf_vectorizer.fit(X_train['combined_text'])

X_train = tfidf_vectorizer.transform(X_train['combined_text'])
X_test = tfidf_vectorizer.transform(X_test['combined_text'])
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1)
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression 
log_model=LogisticRegression(C=1)
log_model.fit(x_train,y_train)


y_train_pred1=log_model.predict(x_train)
print(classification_report(y_train,y_train_pred1))

y_val_pred1=log_model.predict(x_val)
print(classification_report(y_val,y_val_pred1))

y_test_pred1=log_model.predict(X_test)
print(classification_report(y_test,y_test_pred1))

from sklearn.svm import LinearSVC

svc_model=LinearSVC(penalty='l2',C=0.1,dual=False)
svc_model.fit(x_train,y_train)

y_train_pred2=svc_model.predict(x_train)
print(classification_report(y_train,y_train_pred2))

y_val_pred2=svc_model.predict(x_val)
print(classification_report(y_val,y_val_pred2))

y_test_pred2=svc_model.predict(X_test)
print(classification_report(y_test,y_test_pred2))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def predict_genre(title, description, model, vectorizer, label_encoder):

    data = pd.DataFrame({'TITLE': [title], 'DESCRIPTION': [description]})

    data['combined_text'] = data['TITLE'] + ' ' + data['DESCRIPTION'] 

    X_new = vectorizer.transform(data['combined_text'])

    y_pred = model.predict(X_new)

    predicted_genre = label_encoder.inverse_transform(y_pred)[0]

    return predicted_genre

predict_genre("Edgar's Lunch (1998)","L.R. Brane loves his life - his car, his apartment, his job, but especially his girlfriend, Vespa. One day while showering, Vespa runs out of shampoo. L.R. runs across the street to a convenience store to buy some more, a quick trip of no more than a few minutes. When he returns, Vespa is gone and every trace of her existence has been wiped out. L.R.'s life becomes a tortured existence as one strange event after another occurs to confirm in his mind that a conspiracy is working against his finding Vespa.",svc_model,tfidf_vectorizer,le)