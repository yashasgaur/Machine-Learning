import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB

data = pd.read_csv("spam.csv", encoding="latin-1")
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(classification_report(y_test, y_pred))

messages = [
    "Congratulations! You have won a free lottery prize",
    "Hey bro, are we meeting today?",
    "URGENT! Claim your cash reward now",
    "Can you send me the notes from class?"
]

messages_vec = vectorizer.transform(messages)

predictions = model.predict(messages_vec)

for message, prediction in zip(messages, predictions):
    print(message, "->", prediction)

vectorizer = CountVectorizer(binary=True)

X_train_bin = vectorizer.fit_transform(X_train)
X_test_bin = vectorizer.transform(X_test)

model = BernoulliNB()

model.fit(X_train_bin, y_train)

y_pred = model.predict(X_test_bin)

print(accuracy_score(y_test, y_pred))