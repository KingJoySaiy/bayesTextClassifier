#include "BayesClassifier.h"

inline void test(BayesClassifier &classifier, const string &testStr) {

    string data = classifier.test(testStr);
    cout << "\"" << testStr << "\" is " << data << endl;
}
int main() {

    /*
     * 文本预处理：去除中文分词（python jieba)，去除停顿词
     * 使用词频哈希表，不使用“词袋”或“字典树”的原因
     * */

    BayesClassifier classifier("SMSSpamCollection.txt");
    classifier.train();
    classifier.showValid();

    test(classifier, "I am ok");
    test(classifier, "credit Hot");
    test(classifier, "Chicago is a big city");
    test(classifier, "call for premium phone services");

    return 0;
}
