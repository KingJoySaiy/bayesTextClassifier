#include "BayesClassifier.h"

#pragma GCC optimize(3)

void testStringShow(BayesClassifier &classifier) {

    const string strings[] = {"I am ok", "credit Hot", "Chicago is a big city.", "call for premium phone services"};
    cout << endl << "Result of string test:" << endl;
    for (auto &str:strings) {
        cout << "\"" << str << "\" is " << classifier.testString(str) << endl;
    }
}

void testFileShow(BayesClassifier &classifier) {

    const string filePath[] = {"testFile.txt"};
    cout << endl << "Result of file test:" << endl;
    for (auto &path:filePath) {
        auto fileRes = classifier.testFile(path);
        cout << "\"" << fileRes.first << "\" is " << fileRes.second << endl;
    }
}

int main(int argc, char *argv[]) {

    //input trainPath & trainRate
    string trainPath;
    double trainRate;
    cout << "Input trainPath & trainRate(default use \"0 0\") : ";
    cin >> trainPath >> trainRate;
    trainPath = trainPath == "0" ? "SMSSpamCollection.txt" : trainPath;
    trainRate = (trainRate <= 0 or trainRate > 1) ? 0.8 : trainRate;

    //create a classifier, start training
    BayesClassifier classifier(trainPath, trainRate);
    classifier.train();

    //show result of validation
    classifier.showValid();

//    //show test sample
//    testStringShow(classifier);
//    testFileShow(classifier);

    string select, str;
    while (true) {
        cout << "Test your data (str:0, file:1) : ";
        cin >> select;
        if (select != "0" and select != "1") {
            cout << "Invalid select!" << endl;
            break;
        }
        if (select == "0") {
            cout << "Input your strings: " << endl;
            getline(cin, str);
            getline(cin, str);
            cout << classifier.testString(str) << endl;
        } else {
            cout << "Input your path of test file" << endl;
            cin >> str;
            auto res = classifier.testFile(str);
            cout << res.second << ": " << res.first << endl;
        }
    }

    return 0;
}
