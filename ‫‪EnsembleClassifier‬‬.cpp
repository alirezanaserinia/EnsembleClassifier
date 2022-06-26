#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <iomanip>

#define DATA_SET_FILE_PATH "/dataset.csv"
#define LABELS_FILE_PATH "/labels.csv"
#define FILE_TYPE ".csv"
#define PRECISION 2

using namespace std;

struct Class {
    int classifier_number;
    int class_number;
    vector<double> bethas;
    double bias;
};

struct Classifier {
    int classifier_number;
    vector<Class> classes;
};

struct Sample {
    vector<double> features;
    int label;
    vector<int> class_numbers;
    int class_type;
};

typedef vector<Class> Classes;
typedef vector<Classifier> Classifiers;
typedef vector<Sample> Samples;

string read_file_content(const string& file_path) {
	ifstream file(file_path);
	stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

Classes read_classifier(string temp_string, int classifier_number, Classes _classes) {
    stringstream full_stream(temp_string);
    string bethas_names, line;
    int class_number = 0;
    getline(full_stream, bethas_names);
    while(getline(full_stream, line)) {
        stringstream line_stream(line);
        string element;
        Class _class;
        _class.classifier_number = classifier_number;
        _class.class_number = class_number;
        while(getline(line_stream, element, ',')) {
            int next_index_of_element_in_line = line.find(element) + element.length();
            if(line[next_index_of_element_in_line] == ',')
                _class.bethas.push_back(stod(element));      
            else  
                _class.bias = stod(element);
        }
        _classes.push_back(_class);
        class_number++;
    }
    return _classes;
}

Classes read_weight_vectors(const string& file_path) {
    Classes _classes;
    int classifier_number = 0;
    while(true) {
        string file_name = file_path + "/classifier_" + to_string(classifier_number) + FILE_TYPE;
        ifstream file(file_name);
        if(file.good()) {
            string classifier_file_content = read_file_content(file_name);
            _classes = read_classifier(classifier_file_content, classifier_number, _classes);
            classifier_number++;
        }
        else
            break;
    }
    return _classes;
}

Samples read_data_set(string temp_string) {
    Samples samples;
    stringstream full_stream(temp_string);
    string features_names;
    string line;
    getline(full_stream, features_names);
    while(getline(full_stream, line)) {
        stringstream line_stream(line);
        string element;
        Sample sample;
        while(getline(line_stream, element, ',')) {
            sample.features.push_back(stod(element));
        }
        samples.push_back(sample);
    }
    return samples;
}

void set_samples_labels(string labels_file_content, Samples& samples) {
    stringstream full_stream(labels_file_content);
    string labels_names;
    string label;
    getline(full_stream, labels_names);
    for(int sample_number = 0; sample_number < samples.size(); sample_number++) {
        getline(full_stream, label);
        samples[sample_number].label = stoi(label);
    }
}

Samples read_validation_file(string file_path) {
    Samples samples;
    string data_set_file_path = file_path + DATA_SET_FILE_PATH;
    string data_set_file_content = read_file_content(data_set_file_path);
    samples = read_data_set(data_set_file_content);
    string labels_file_path = file_path + LABELS_FILE_PATH;
    string labels_file_content = read_file_content(labels_file_path);
    set_samples_labels(labels_file_content, samples);
    return samples;
}

int find_number_of_classifiers(Classes classes) {
    int max = 0;
    for(int class_number = 0; class_number < classes.size(); class_number++) {
        if(classes[class_number].classifier_number > max)
            max = classes[class_number].classifier_number;
    }
    return max + 1;
}

Classifiers set_classifiers(Classes classes) {
    Classifiers classifiers;
    int number_of_classifiers = find_number_of_classifiers(classes);
    for(int classifier_number = 0; classifier_number < number_of_classifiers; classifier_number++) {
        Classifier classifier;
        classifier.classifier_number = classifier_number;
        for(int class_number = 0; class_number < classes.size(); class_number++) {
            if(classes[class_number].classifier_number == classifier_number)
                classifier.classes.push_back(classes[class_number]);
        }
        classifiers.push_back(classifier);
    }
    return classifiers;
}

int specify_most_score(vector<double> scores) {
    int most_score_index = 0;
    for(int score_number = 1; score_number < scores.size(); score_number++) {
        if(scores[score_number] > scores[score_number - 1])
            most_score_index = score_number;
    }
    return most_score_index;
}

int specify_class_type(Sample sample, Classifier classifier) {
    vector<double> scores;
    for(int class_number = 0; class_number < classifier.classes.size(); class_number++) {
        double multiplication = 0;
        for(int index = 0; index < sample.features.size(); index++)
            multiplication += sample.features[index] * classifier.classes[class_number].bethas[index];
        double score = multiplication + classifier.classes[class_number].bias;
        scores.push_back(score);
    }
    int class_type = specify_most_score(scores);
    return class_type;
}

Samples samples_linear_classification(Samples samples, Classifiers classifiers) {
    for(int sample_number = 0; sample_number < samples.size(); sample_number++) {
        for(int classifier_number = 0; classifier_number < classifiers.size(); classifier_number++) {
            int class_type = specify_class_type(samples[sample_number], classifiers[classifier_number]);
            samples[sample_number].class_numbers.push_back(class_type);
        }
    }
    return samples;
}

int find_max_index_of_list(vector<int> list) {
    int max_index = 0;
    for(int index = 1; index < list.size(); index++) {
        if(list[index] > list[index - 1])
            max_index = index;
    }
    return max_index;
}

Samples samples_ensemble_classification(Samples samples) {
    for(int sample_number = 0; sample_number < samples.size(); sample_number++) {
        const int number_of_classes = samples[sample_number].class_numbers.size();
        vector<int> class_numbers(number_of_classes, 0);
        for(int index = 0; index < samples[sample_number].class_numbers.size(); index++) {
            class_numbers[samples[sample_number].class_numbers[index]]++;
        }
    samples[sample_number].class_type = find_max_index_of_list(class_numbers);
    }
    return samples;
}

double compare_samples_class_type_with_labels(Samples samples) {
    int number_of_correct_predictions = 0;
    for(int sample_number = 0; sample_number < samples.size(); sample_number++) {
        if(samples[sample_number].class_type == samples[sample_number].label)
            number_of_correct_predictions++;
    }
    return number_of_correct_predictions * (100/(double)samples.size());
}

void print_accuracy(Samples samples) {
    double accuracy = compare_samples_class_type_with_labels(samples);
    cout << "Accuracy: ";
    cout << fixed << setprecision(PRECISION) << accuracy << "%" << endl;
}

int main(int argc, char* argv[]) {
    Samples samples = read_validation_file(argv[1]);
    Classes classes = read_weight_vectors(argv[2]);
    Classifiers classifiers = set_classifiers(classes);
    samples = samples_linear_classification(samples, classifiers);
    samples = samples_ensemble_classification(samples);
    print_accuracy(samples);
    return 0;
}