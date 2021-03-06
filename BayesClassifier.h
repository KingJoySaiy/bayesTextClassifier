//
// Created by Joy on 2020/12/13
//

#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include <bits/stdc++.h>

using namespace std;

inline vector<string> getFeatures(const string &strings, unordered_set<string> &stopWords) { //erase illegal characters

    stringstream ss(strings);
    string tmp;
    vector<string> res;
    while (ss >> tmp) {
        //erase non-digit & non-alpha at back
        while (!tmp.empty() and !isdigit(tmp.back()) and !isalpha(tmp.back())) {
            tmp.pop_back();
        }
        //erase non-digit & non-alpha at front
        while (!tmp.empty() and !isdigit(tmp[0]) and !isalpha(tmp[0])) {
            tmp.erase(tmp.begin());
        }
        //skip empty & stop words
        if (!tmp.empty() and stopWords.find(tmp) == stopWords.end()) {
            res.push_back(tmp);
        }
    }
    return res;
}

class BayesClassifier {
private:
    vector<pair<vector<string>, string>> trainData, validData;    //(strings, className)
    unordered_set<string> stopWords;    //English stop word
    map<string, double> classProb;   //P(class), class -> prob
    map<string, int> classSize; //unique strings in each class
    map<string, map<string, int>> strC;    //counts of pair(class, str), class -> (str->count)
    int strCount = 0;   //size of unique strings

    inline double getLogClassProb(const vector<string> &strings, const string &className) {

        double res = log(classProb[className]);
        for (auto &str:strings) {
            res += log(double(strC[className][str] + 1) / (classSize[className] + strCount));
        }
        return res;
    }

public:
    //constructer, load English stop word ,training set & validation set
    explicit BayesClassifier(const string &trainPath = "SMSSpamCollection.txt", const double trainRate = 0.8) {

        //load English stop word
        ifstream file("EnglishStopWord.txt");
        string line, tmp;
        if (!file.is_open()) {
            cout << "Fail to read \"EnglishStopWord.txt\"!" << endl;
            exit(0);
        }
        while (getline(file, line)) {
            stopWords.insert(line);
        }
        file.close();

        //load training set
        file.open(trainPath);
        if (!file.is_open()) {
            cout << "Fail to read training file!" << endl;
            exit(0);
        }
        while (getline(file, line)) {
            auto data = getFeatures(line, stopWords);
            tmp = data.front();
            data.erase(data.begin());
            trainData.emplace_back(data, tmp);
        }
        file.close();

        //load validation set
        int trainSize = int(trainRate * trainData.size());
        while (trainData.size() > trainSize) {
            validData.push_back(trainData.back());
            trainData.pop_back();
        }
    }

    void train() {  //start training
        //calculate P(class)
        classProb.clear();
        map<string, int> classCount;    //classNames -> count
        for (auto &p : trainData) {
            classCount[p.second]++;
        }
        for (auto &p : classCount) {
            classProb[p.first] = double(p.second) / trainData.size();
        }

        //count pair(class, str)
        strC.clear();
        set<string> uniqueStr;
        for (auto &data:trainData) {
            auto &strings = data.first;
            auto &Class = data.second;
            set<string> book;   //count str once

            for (auto &str:strings) {
                uniqueStr.insert(str);
                book.insert(str);
            }
            for (auto &str:book) {  //str only count once
                strC[Class][str]++;
            }
        }
        strCount = uniqueStr.size();

        //count unique strings in each class
        classSize.clear();
        for (auto &p : strC) {
            classSize[p.first] = p.second.size();
        }
    }

    void showValid() {  //show result of validation set
        auto start = clock();
        int correct = 0;
        for (auto& data : validData) {
            double maxProb = -DBL_MAX, nowProb;
            string resClass;
            for (auto &p:classSize) {
                auto &className = p.first;
                nowProb = getLogClassProb(data.first, className);
                if (nowProb > maxProb) {
                    maxProb = nowProb;
                    resClass = className;
                }
            }
            if (resClass == data.second) {
                correct++;
            }
        }
        cout << "Accuracy of validation: " << correct << " / " << validData.size() << " = "<< double(correct) / validData.size() << endl;
        auto end = clock();
        cout << "Validation time:" << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    }

    string testString(const string &testData) { //test strings

        auto strings = getFeatures(testData, stopWords);
        double maxProb = -DBL_MAX, nowProb;
        string resClass;
        for (auto &data:classSize) {
            auto &className = data.first;
            nowProb = getLogClassProb(strings, className);
            if (nowProb > maxProb) {
                maxProb = nowProb;
                resClass = className;
            }
        }
        return resClass;
    }

    pair<string, string> testFile(const string &testPath) { //test a file

        ifstream file(testPath);
        string line, res;
        if (!file.is_open()) {
            cout << "Fail to read test file!" << endl;
            exit(0);
        }
        while (getline(file, line)) {
            res += line + " ";
        }
        file.close();
        return make_pair(res, testString(res));
    }
};

#endif //BAYESCLASSIFIER_H
