#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Mueller / mathias.mueller@uzh.ch
# Edited: Albina Kudoyarova / albina.kudoyarova@uzh.ch
# LALA at the Kaggle.com

from __future__ import unicode_literals

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from collections import defaultdict

import logging
import argparse
import random
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf8')

random.seed(42)

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


class Trainer(object):
    """
    Reads raw dialect data and trains a classifier.
    """

    def __init__(self, model="model.pkl", data=None, verbose=False,
    	classifier=None):
        """
        """
        self._model = model
        self._data = data
        self._verbose = verbose
        self._classifier = classifier
        # outcomes
        self.classes = []
        self.num_classes = 0
        self.train_X = None
        self.train_y = None
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None

    def train(self):
        """
        Preprocesses data, fits a model, and finally saves the model to a file.
        """
        self._preprocess()
        self._build_pipeline()
        self._fit()

    def _preprocess(self):
        """
        Reads lines from the raw dialect data.
        """
        d = defaultdict(list)

        if self._data:
            data = codecs.open(self._data, "r", "UTF-8")
        else:
            logging.debug("--data not found, assuming input from STDIN")
            data = sys.stdin

        # read first line with column identifiers and ignore
        data.readline()

        for line in data:
            # skip empty lines
            line = line.strip()
            if line == "":
                continue

            X, y = line.split(",")
            d[y].append(X)

        logging.debug("Examples per dialect class:")
        for k, v in d.iteritems():
            logging.debug("%s %d" % (k, len(v)))
        logging.debug("Total messages: %d\n" %
                      sum([len(v) for v in d.values()]))

        self.classes = d.keys()
        self.classes.sort()
        self.num_classes = len(self.classes)

        l = []
        logging.debug("Samples from the data:")
        for k, values in d.iteritems():
        	logging.debug("%s\t%s" % (values[0], k))
        	for value in values:
        		l.append( (value, k) )

        # shuffle, just to be sure
        random.shuffle(l)
        self.train_X, self.train_y = zip(*l)

    def _build_pipeline(self):
        """
        Builds an sklearn Pipeline. The pipeline consists of a kind of
        vectorizer, followed by a kind of classifier.
        """
        # strip_accents="ascii" is a bad idea, results in lower accuracy due to the fact, that specific chars are used in specific dialects
        # analyzer="words" is a bad idea,  results in lower accuracy
        self.vectorizer = TfidfVectorizer(ngram_range=(1,6), analyzer='char')
        self.classifier = MLPClassifier(verbose=True, early_stopping=False) # TODO: early stopping?

        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("clf", self.classifier)
        ])

        logging.debug(self.vectorizer)
        logging.debug(self.classifier)
        logging.debug(self.pipeline)

    def _fit(self):
        """
        Fits a model for the preprocessed data.
        """
        self.pipeline.fit(self.train_X, self.train_y)

    def save(self):
        """
        Save the whole pipeline to a pickled file.
        """
        from sklearn.externals import joblib
        joblib.dump(self.pipeline, self._model)
        logging.debug("Classifier saved to '%s'" % self._model)


class Predictor(object):
    """
    Predicts the dialect of text, given a trained model.
    """

    def __init__(self, model="model.pkl"):
        """
        """
        self._model = model
        self._load()

    def _load(self):
        """
        Loads a model that was previously trained and saved.
        """
        from sklearn.externals import joblib
        self.pipeline = joblib.load(self._model)
        logging.debug("Loading model pipeline from '%s'" % self._model)

    def predict(self, samples, label_only=False):
        """
        Predicts the class (=dialect) of new text samples.
        """
        predictions = []

        for sample in samples:
            if label_only:
            	predictions.append(self.pipeline.predict([sample])[0])
            else:
                sample = sample.strip().split(",")[1]  # column 0 is the index
            	predictions.append((sample, self.pipeline.predict([sample])[0]))

        return predictions

    def evaluate(self, samples):
    	"""
    	Evaluates the classifier with gold labelled data.
    	"""
    	test_y = []
    	test_X = []
    	for sample in samples:
    		sample = sample.strip()
    		X,y = sample.split(",")
    		test_y.append(y)
    		test_X.append(X)

    	logging.debug("Number of gold samples found: %d" % len(test_y))

    	predictions = self.predict(test_X, label_only=True)
        report = metrics.classification_report(test_y, predictions, target_names=None)
        logging.info(report)


def parse_cmd():
    parser = argparse.ArgumentParser(
        description="train a classifier for dialect data and use it for predictions")

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=False,
        help="if --train, then save model to this path. If --predict, use saved model at this path."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        required=False,
        help="write verbose output to STDERR (default: False)"
    )

    mode_options = parser.add_mutually_exclusive_group(required=True)
    mode_options.add_argument(
        "--train",
        action="store_true",
        required=False,
        help="train a new model and save to the path -m/--model"
    )
    mode_options.add_argument(
        "--predict",
        action="store_true",
        required=False,
        help="predict classes of new samples, write predicted classes to STDOUT"
    )
    mode_options.add_argument(
        "--evaluate",
        action="store_true",
        required=False,
        help="evaluate trained model, write report to STDOUT. If --evaluate, data in --samples is assumed to include the gold label"
    )
    mode_options.add_argument(
        "--split",
        action="store_true",
        required=False,
        help="split data from --data to train and test sets and then exit"
    )

    train_options = parser.add_argument_group("training parameters")

    train_options.add_argument(
        "--data",
        type=str,
        required=False,
        help="path to file with raw dialect data, UTF-8. If --data is not given, input from STDIN is assumed"
    )
    train_options.add_argument(
        "--classifier",
        type=str,
        required=False,
        default="mlp",
        help="type of classifier to be trained. Either 'mlp' or 'dummy' (stratified class probabilities)",
        choices=("mlp", "dummy")
    )

    predict_options = parser.add_argument_group("prediction parameters")

    predict_options.add_argument(
        "--samples",
        type=str,
        required=False,
        help="Path to file containing samples for which a class should be predicted. If --samples is not given, input from STDIN is assumed"
    )

    split_options = parser.add_argument_group("split parameters")

    split_options.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        required=False,
        help="flag to force data shuffling when splitting"
    )

    split_options.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        required=False,
        help="a number between 0 and 1 to specify the percentage of a test subset, e.g. 0.1 would mean 10 percent i.e. 90/10 split"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_cmd()

    # set up logging
    if args.verbose:
        level = logging.DEBUG
    elif args.evaluate:
    	level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    if args.split:
        logging.debug("splitting file %s into 2 files" % args.data)

        lines = []
        if args.data:
            with codecs.open(args.data, 'r', encoding="utf-8") as f:
                lines = f.readlines()

            header_line = lines[0]
            res = train_test_split(lines[1:], shuffle=args.shuffle, test_size=args.test_size)

            with codecs.open("{}.split{}".format(args.data, (int)((1 - args.test_size)*100)), 'w', encoding="utf-8") as out:
                out.write(header_line)
                out.writelines(res[0])

            with codecs.open("{}.split{}".format(args.data, (int)(args.test_size*100)), 'w', encoding="utf-8") as out:
                out.write(header_line)
                out.writelines(res[1])

        #splitting is a separate step, so we stop after it is done
        return

    if args.train:
        t = Trainer(model=args.model,
                    data=args.data,
                    verbose=args.verbose,
                    classifier=args.classifier
                    )
        t.train()
        t.save()
    else:
        p = Predictor(model=args.model)
        if args.samples:
            input_ = codecs.open(args.samples, "r", "UTF-8")
        else:
            logging.debug("--samples not found, assuming input from STDIN")
            input_ = sys.stdin

        # read first line and ignore, column names
        input_.readline()

        if args.evaluate:
        	p.evaluate(samples=input_)
        else:
            predictions = p.predict(samples=input_, label_only=True)
            print "Id,Prediction"
            for index, prediction in enumerate(predictions):
        		print "%s,%s" % (index+1, prediction)


if __name__ == '__main__':
    main()
