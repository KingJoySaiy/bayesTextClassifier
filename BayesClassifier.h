//
// Created by Joy on 2020/12/13
//

#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include <bits/stdc++.h>

//using namespace std;
using std::vector;
using std::string;
using std::map;
using std::pair;
using std::set;
using std::ifstream;
using std::istringstream;

class BayesClassifier {
private:
    vector<pair<vector<string>, string>> trainData;    //(strings, className)
    map<string, double> classProb;   //P(class), class -> prob
    map<string, int> classSize; //unique strings in each class
    map<string, map<string, int>> strC;    //counts of pair(class, str), class -> (str->count)
    int strCount = 0;

    double getLogClassProb(const vector<string> &strings, const string &className) {

        double res = log(classProb[className]);
        for (auto &str:strings) {
            res += log(double(strC[className][str] + 1) / (classSize[className] + strCount));
        }
        return res;
    }
public:
    explicit BayesClassifier(const string &trainPath) {

        //save training data
        ifstream file(trainPath);
        string line, tmp;
        while (getline(file, line)) {
            istringstream stream(line);
            vector<string> data;
            while (stream >> tmp) {
                data.push_back(tmp);
            }
            tmp = data.back();
            data.pop_back();
            trainData.emplace_back(data, tmp);
        }
    }

    void train() {
        //calculate P(class)
        map<string, int> classCount;    //classNames -> count
        for (auto &p : trainData) {
            classCount[p.second]++;
        }
        for (auto &p : classCount) {
            classProb[p.first] = double(p.second) / trainData.size();
        }

        //count pair(class, str)
        set<string> distStr;
        for (auto &data:trainData) {
            auto &strings = data.first;
            auto &Class = data.second;
            set<string> book;   //count str once

            for (auto &str:strings) {
                distStr.insert(str);
                book.insert(str);
            }
            for (auto &str:book) {  //str only count once
                strC[Class][str]++;
            }
        }
        strCount = distStr.size();
//        cout << "Total unique attributes count: " << strCount << endl;

        //count unique strings in each class
        for (auto &p : strC) {
            classSize[p.first] = p.second.size();
//            cout << p.first << ": " << classSize[p.first] << endl;
        }
    }

    string test(const string &testData) {

        istringstream stream(testData);
        vector<string> strings;
        string tmp;
        while (stream >> tmp) {
            strings.push_back(tmp);
        }

        set<string> book;   //each class only calculate once
        double maxProb = -DBL_MAX, nowProb;
        string resClass;
        for (auto &data:trainData) {
            auto &className = data.second;
            if (book.find(className) != book.end()) continue;
            nowProb = getLogClassProb(strings, className);
            if (nowProb > maxProb) {
                maxProb = nowProb;
                resClass = className;
            }
            book.insert(className);
        }
        return resClass;
    }
};

#endif //BAYESCLASSIFIER_H
