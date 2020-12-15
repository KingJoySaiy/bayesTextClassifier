#include "BayesClassifier.h"

using std::cout;
using std::endl;

inline void test(BayesClassifier &classifier, const string &testStr) {

    string data = classifier.test(testStr);
    cout << "\"" << testStr << "\" is " << data << endl;
}
int main() {

    BayesClassifier classifier("training.txt");
    classifier.train();

    test(classifier, "I am ok");
    test(classifier, "credit Hot");
    test(classifier, "Chicago is a big city");
    test(classifier, "king of sadness of try");

    return 0;
}
