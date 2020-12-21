import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction import text
import time

# load training file
df = pd.read_csv("SMSSpamCollection.txt", delimiter='\t', header=None)
trainSize, validSize = int(len(df) * 0.8), len(df) - int(len(df) * 0.8)

# training
y, X = df[0][:trainSize], df[1][:trainSize]
vectorizer = text.TfidfVectorizer()   # vectorization
# vectorizer = text.HashingVectorizer()
X = vectorizer.fit_transform(X)
model = MultinomialNB()
# model = BernoulliNB()


model.fit(X, y)

# validation
start = time.perf_counter()
string, label = df[1].to_list(), df[0].to_list()
testX = vectorizer.transform(df[1][-validSize:])
predictions = list(model.predict(testX))
ok = 0
for i in range(validSize):
    if predictions[i] == label[-validSize + i]:
        ok += 1
print("Accuracy of validation: ", ok, "/", validSize, "=", ok / validSize)
print("Validation time:", time.perf_counter() - start)