#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_map>

using namespace std;

// Define a struct to represent a single data point in the CSV file
struct DataPoint {
    vector<double> features;
    double label;
};

// Define a struct to represent a decision node in the tree
struct Node {
    int feature_index;
    double threshold;
    bool is_leaf;
    double value;
    Node* left_child;
    Node* right_child;

    Node() {
        feature_index = -1;
        threshold = 0.0;
        is_leaf = false;
        value = 0.0;
        left_child = nullptr;
        right_child = nullptr;
    }

    ~Node() {
        delete left_child;
        delete right_child;
    }
};

// Define a struct to represent a decision tree
struct DecisionTree {
    Node* root;

    DecisionTree() {
        root = nullptr;
    }

    ~DecisionTree() {
        delete root;
    }
};

// Define a struct to represent a forest of decision trees
struct RandomForest {
    vector<DecisionTree*> trees;

    RandomForest() {}

    ~RandomForest() {
        for (auto tree : trees) {
            delete tree;
        }
    }
};

// Helper function to split a string into a vector of substrings using a delimiter
vector<string> split(const string& str, char delimiter) {
    vector<string> tokens;
    stringstream ss(str);
    string token;
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Helper function to read in a CSV file and convert it to a vector of DataPoints
vector<DataPoint> read_csv(string filename) {
    vector<DataPoint> data;
    ifstream file(filename);
    if (file.is_open()) {
        string line;
        getline(file, line); // Skip header line
        while (getline(file, line)) {
            vector<string> tokens = split(line, ',');
            DataPoint point;
            for (int i = 0; i < tokens.size() - 1; i++) {
                point.features.push_back(stod(tokens[i]));
            }
            point.label = stod(tokens[tokens.size() - 1]);
            data.push_back(point);
        }
        file.close();
    }
    return data;
}

// Helper function to calculate the mean of a vector of doubles
double mean(const vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    return sum / v.size();
}

// Helper function to calculate the variance of a vector of doubles
double variance(const vector<double>& v) {
    double m = mean(v);
    double sum = 0.0;
    for (double x : v) {
        sum += (x - m) * (x - m);
    }
    return sum / (v.size() - 1);
}

// Helper function to calculate the entropy of a vector of doubles
double entropy(const vector<double>& v) {
    double e = 0.0;
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    for (double x : v) {
        if (x > 0) {
            double p = x / sum;
            e -=p * log2(p);
    }
}
return e;}

// Helper function to split a dataset into two subsets based on a given feature and threshold
pair<vector<DataPoint>, vector<DataPoint>> split_dataset(const vector<DataPoint>& data, int feature_index, double threshold) {
vector<DataPoint> left, right;
for (const DataPoint& point : data) {
if (point.features[feature_index] < threshold) {
left.push_back(point);
} else {
right.push_back(point);
}
}
return make_pair(left, right);
}
// Helper function to select a random subset of features to use when building a tree
vector<int> random_features(int num_features, int max_features) {
vector<int> indices(num_features);
for (int i = 0; i < num_features; i++) {
indices[i] = i;
}
random_shuffle(indices.begin(), indices.end());
indices.resize(max_features);
sort(indices.begin(), indices.end());
return indices;
}

// Helper function to build a decision tree using a recursive algorithm
Node* build_tree(const vector<DataPoint>& data, const vector<int>& feature_indices, int max_depth, int min_samples_split) {
// Create a leaf node if the dataset is small enough or the maximum depth has been reached
if (data.size() <= min_samples_split || max_depth == 0) {
Node* node = new Node();
node->is_leaf = true;
node->value = mean({point.label for (const DataPoint& point : data)});
return node;
}
// Choose the best feature and threshold to split the dataset
int best_feature = 0;
double best_threshold = 0.0;
double best_score = INFINITY;
for (int feature_index : feature_indices) {
vector<double> values;
for (const DataPoint& point : data) {
values.push_back(point.features[feature_index]);
}
sort(values.begin(), values.end());
for (int i = 1; i < values.size(); i++) {
double threshold = (values[i] + values[i - 1]) / 2;
auto [left, right] = split_dataset(data, feature_index, threshold);
if (left.empty() || right.empty()) {
continue;
}
double score = entropy({left.size(), right.size()}) - ((left.size() / data.size()) * entropy({left.size(), right.size()})) - ((right.size() / data.size()) * entropy({left.size(), right.size()}));
if (score < best_score) {
best_feature = feature_index;
best_threshold = threshold;
best_score = score;
}
}
}
// Create a decision node and recursively build the left and right subtrees
auto [left, right] = split_dataset(data, best_feature, best_threshold);
Node* node = new Node();
node->feature_index = best_feature;
node->threshold = best_threshold;
node->left_child = build_tree(left, feature_indices, max_depth - 1, min_samples_split);
node->right_child = build_tree(right, feature_indices, max_depth - 1, min_samples_split);
return node;
}

// Helper function to make a prediction using a decision tree
double predict(const Node* node, const DataPoint& point) {
if (node->is_leaf) {
return node->value;
}
if (point.features[node->feature_index] < node->threshold) {
return predict(node->left_child, point);
} else {
return predict(node->right




1052 - Nagumantry Raghuveer
finish
}

// Class definition for the Random Forest model
class RandomForest {
public:
// Constructor to initialize the model parameters
RandomForest(int num_trees, int max_depth, int min_samples_split, int max_features) :
num_trees_(num_trees),
max_depth_(max_depth),
min_samples_split_(min_samples_split),
max_features_(max_features) {}

arduino
Copy code
// Function to train the model on a given dataset
void fit(const vector<DataPoint>& data) {
    int num_features = data[0].features.size();
    for (int i = 0; i < num_trees_; i++) {
        // Randomly select a subset of the features to use for this tree
        vector<int> feature_indices = random_features(num_features, max_features_);
        // Randomly sample a subset of the data to use for this tree
        vector<DataPoint> sample;
        for (int j = 0; j < data.size(); j++) {
            int random_index = rand() % data.size();
            sample.push_back(data[random_index]);
        }
        // Build the decision tree using the selected features and data subset
        trees_.push_back(build_tree(sample, feature_indices, max_depth_, min_samples_split_));
    }
}

// Function to make a prediction on a given data point
double predict(const DataPoint& point) {
    vector<double> predictions;
    for (const Node* tree : trees_) {
        predictions.push_back(::predict(tree, point));
    }
    return mean(predictions);
}
private:
int num_trees_;
int max_depth_;
int min_samples_split_;
int max_features_;
vector<Node*> trees_;
};

// Function to read a CSV file and create a vector of DataPoints
vector<DataPoint> read_data(const string& filename) {
vector<DataPoint> data;
ifstream file(filename);
string line;
while (getline(file, line)) {
stringstream ss(line);
string value;
DataPoint point;
while (getline(ss, value, ',')) {
point.features.push_back(stod(value));
}
point.label = point.features.back();
point.features.pop_back();
data.push_back(point);
}
return data;
}

int main() {
// Read the training data from a CSV file
vector<DataPoint> data = read_data("train.csv");
// Train a Random Forest model on the training data
RandomForest model(10, 5, 2, 4);
model.fit(data);
// Read the test data from a CSV file
vector<DataPoint> test_data = read_data("test.csv");
// Make predictions on the test data and write the results to a CSV file
ofstream file("predictions.csv");
for (const DataPoint& point : test_data) {
double prediction = model.predict(point);
file << prediction << endl;
}
file.close();
return 0;
}
// Choose the best feature and threshold to split the dataset
int best_feature = 0;
double best_threshold = 0.0;
double best_score = INFINITY;
for (int feature_index : feature_indices) {
vector<double> values;
for (const DataPoint& point : data) {
values.push_back(point.features[feature_index]);
}
sort(values.begin(), values.end());
for (int i = 1; i < values.size(); i++) {
double threshold = (values[i] + values[i - 1]) / 2;
auto [left, right] = split_dataset(data, feature_index, threshold);
if (left.empty() || right.empty()) {
continue;
}
double score = entropy({left.size(), right.size()}) - ((left.size() / data.size()) * entropy({left.size(), right.size()})) - ((right.size() / data.size()) * entropy({left.size(), right.size()}));
if (score < best_score) {
best_feature = feature_index;
best_threshold = threshold;
best_score = score;
}
}
}
// Create a decision node and recursively build the left and right subtrees
auto [left, right] = split_dataset(data, best_feature, best_threshold);
Node* node = new Node();
node->feature_index = best_feature;
node->threshold = best_threshold;
node->left_child = build_tree(left, feature_indices, max_depth - 1, min_samples_split);
node->right_child = build_tree(right, feature_indices, max_depth - 1, min_samples_split);
return node;
}

// Helper function to make a prediction using a decision tree
double predict(const Node* node, const DataPoint& point) {
if (node->is_leaf) {
return node->value;
}
if (point.features[node->feature_index] < node->threshold) {
return predict(node->left_child, point);
} else {
return predict(node->_child, point);
}
}

// Class definition for the Random Forest model
class RandomForest {
public:
RandomForest(int num_trees, int max_depth, int min_samples_split, int max_features);
void fit(const vector<DataPoint>& data);
vector<double> predict(const vector<DataPoint>& data);

private:
int num_trees_;
int max_depth_;
int min_samples_split_;
int max_features_;
vector<Node*> trees_;
};

// Constructor for the Random Forest model
RandomForest::RandomForest(int num_trees, int max_depth, int min_samples_split, int max_features)
: num_trees_(num_trees), max_depth_(max_depth), min_samples_split_(min_samples_split), max_features_(max_features) {}

// Function to fit the Random Forest model to a dataset
void RandomForest::fit(const vector<DataPoint>& data) {
int num_features = data[0].features.size();
for (int i = 0; i < num_trees_; i++) {
// Select a random subset of features to use for this tree
vector<int> feature_indices = random_features(num_features, max_features_);
// Select a random subset of the data points to use for this tree
vector<DataPoint> subset;
for (int j = 0; j < data.size(); j++) {
if (rand() / (double)RAND_MAX < 0.5) {
subset.push_back(data[j]);
}
}
// Build the decision tree for this subset of the data
Node* tree = build_tree(subset, feature_indices, max_depth_, min_samples_split_);
trees_.push_back(tree);
}
}

// Function to make predictions using the Random Forest model
vector<double> RandomForest::predict(const vector<DataPoint>& data) {
vector<double> predictions(data.size());
for (int i = 0; i < data.size(); i++) {
double sum = 0.0;
for (Node* tree : trees_) {
sum += predict(tree, data[i]);
}
predictions[i] = sum / trees_.size();
}
return predictions;
}

// Helper function to read a CSV file and convert it to a vector of DataPoints
vector<DataPoint> read_csv(string filename) {
vector<DataPoint> data;
ifstream file(filename);
string line;
getline(file, line); // discard header line
while (getline(file, line)) {
stringstream ss(line);
string field;
vector<string> fields;
while (getline(ss, field, ',')) {
fields.push_back(field);
}
DataPoint point;
point.label = stod(fields.back());
fields.pop_back();
for (const string& value : fields) {
point.features.push_back(stod(value));
}
data.push_back(point);
}
return data;
}

// Helper function to write a vector of predictions to a CSV file
void write_csv(string filename, const vector<double>& predictions) {
ofstream file(filename);
for (double prediction : predictions) {
file << prediction << "\n";
}
}

// Main function to run the Random Forest model on a CSV file
int main(int argc, char** argv) {
if (argc != 6) {
cout << "Usage: ./random_forest <csv_filename> <num_trees> <max_depth> <min_samples_split> <max_features>\n";
return 1;
}
string filename = argv[1];
int num_trees = stoi(argv[2]);
int max_depth = stoi(argv[3]);
int min_samples_split = stoi(argv[4]);
int max_features = stoi(argv[5]);
srand(time(NULL)); // Seed the random number generator

vector<DataPoint> data = read_csv();

RandomForest model(num_trees, max_depth, min_samples_split, max_features);
model.fit(data);

vector<double> predictions = model.predict(data);

write_csv("predictions.csv", predictions);

return 0;
}







