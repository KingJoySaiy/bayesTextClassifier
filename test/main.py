import pandas as pd
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.feature_extraction import text
import time

# load training file
filePath = "SMSSpamCollection.txt"
df = pd.read_csv(filePath, delimiter='\t', header=None)
trainSize = int(len(df) * 0.8)
validSize = len(df) - trainSize

# get X & y , then vectorization
y, X = df[0][:trainSize], df[1][:trainSize]
vectorizer = text.TfidfVectorizer()
X = vectorizer.fit_transform(X)

# use Bernoulli model
model = BernoulliNB()
# model = MultinomialNB()
model.fit(X, y)

# validation
start = time.perf_counter()
string, label = df[1].to_list(), df[0].to_list()
testX = vectorizer.transform(df[1][-validSize:])
predictions = model.predict(testX)
ok = 0
for i in range(validSize):
    if predictions[i] == label[-validSize + i]:
        ok += 1
print("Accuracy of validation: ", ok, "/", validSize, "=", ok / validSize)
print("Validation time:", time.perf_counter() - start)