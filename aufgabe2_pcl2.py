# !/usr/bin/env python
#  -*- coding: utf-8 -*-
# PCL II, FS 17
# Uebung 5, Aufgabe 2
# Autor: Luca Salini, Albina Kudoyarova
# Matrikel-Nr.: 16-732-505, 15-703-390

"""This module uses a naive bayes classifier to identify the gender of names"""

import random
import collections
from nltk.metrics import precision, recall, f_measure
from nltk.classify import NaiveBayesClassifier, accuracy


class NaiveBayesClassifierNameGenderPrediction:
    """This class implements the naive bayes classification on gender
    recognition of names"""

    def __init__(self, female_file, male_file):
        self.male_file = male_file
        self.female_file = female_file

        self.main()

    @staticmethod
    def extract_data(data_file):
        """extract the given data file"""
        extracted_file = open(data_file, 'r', encoding='utf-8')
        extracted_file = extracted_file.read()
        return extracted_file

    @staticmethod
    def evaluation(test_set, classifier):
        """Evaluate the classifier with the test set. Print the accuracy,
        precision, recall and f-measure."""

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for i, (featureset, label) in enumerate(test_set):
            refsets[label].add(i)
            observed = classifier.classify(featureset)
            testsets[observed].add(i)

        print('Evaluation Results')
        print("\t\t\t{:<20}{:0.2f}".format('classifier accuracy', accuracy(classifier, test_set)))

        print("\t\t\t{:<20}{:0.2f}".format('precision male', precision(refsets['male'], testsets['male'])))
        print("\t\t\t{:<20}{:0.2f}".format('precision female', precision(refsets['female'], testsets['female'])))

        print("\t\t\t{:<20}{:0.2f}".format('recall male', recall(refsets['male'], testsets['male'])))
        print("\t\t\t{:<20}{:0.2f}".format('recall female', recall(refsets['female'], testsets['female'])))

        print("\t\t\t{:<20}{:0.2f}".format('f_measure male', f_measure(refsets['male'], testsets['male'])))
        print("\t\t\t{:<20}{:0.2f}".format('f_measure female', f_measure(refsets['female'], testsets['female'])))
        print()

    @staticmethod
    def gender_features(name):
        """Return a dictionary with all features to identify the gender of a name"""
        return {
            'first_char': name[0],
            'first_2_chars': name[:2],

            'last_char': name[-1],
            'last_2_chars': name[-2:],
        }

    def get_training_and_test_labeled_features(self, female_training_data,
                                               male_training_data,
                                               female_test_data,
                                               male_test_data):
        """return a labeled dictionary of all features for the training and test data"""
        # Each list should be in the form [({feature1_name:feature1_value, ...}, label), ...]

        train = [(self.gender_features(x), 'female') for x in female_training_data] + \
                [(self.gender_features(x), 'male') for x in male_training_data]

        test = [(self.gender_features(x), 'female') for x in female_test_data] + \
               [(self.gender_features(x), 'male') for x in male_test_data]

        return train, test

    @staticmethod
    def get_train_and_test_data(male_data, female_data):
        """Split the male and female data into training and test data."""

        res = NaiveBayesClassifierNameGenderPrediction.split_into_test_and_train(male_data) + \
              NaiveBayesClassifierNameGenderPrediction.split_into_test_and_train(female_data)
        return res

    @staticmethod
    def split_into_test_and_train(input_data):
        """Split data list into test and train data lists"""
        data_list = input_data.split('\n')
        # filter out empty strings
        data_list = list(filter(None, data_list))

        random.shuffle(data_list)

        # how many % of data to put into train, should be between 1 and 99
        train_percentage = 80
        separator_index = int(len(data_list) / 100 * train_percentage)

        res_train = data_list[:separator_index]
        res_test = data_list[separator_index:]

        return res_train, res_test

    def main(self):
        """ main method """
        male_data = self.extract_data(self.male_file)
        female_data = self.extract_data(self.female_file)

        male_train_data, male_test_data, female_train_data, female_test_data = \
            self.get_train_and_test_data(male_data, female_data)

        # get the training and test set for the classifier and the evaluation
        train_set, test_set = self.get_training_and_test_labeled_features(
            female_train_data, male_train_data, female_test_data,
            male_test_data)

        # create classifier with the training set
        classifier = NaiveBayesClassifier.train(train_set)

        # print the evaluation with the precision, recall and f-measure
        self.evaluation(test_set, classifier)

        # print the 10 most informative features
        classifier.show_most_informative_features(10)

if __name__ == '__main__':
    NaiveBayesClassifierNameGenderPrediction('female.txt', 'male.txt')
