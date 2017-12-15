# swiss-dialect-identification
Baseline implementation for swiss dialect identification on the VarDial 2017 data set. Please use Python 2.X to run the script.

## Installation and Requirements

The script requires a recent version of the `scikit-learn` package. In most cases, installation is as easy as

    pip install scikit-learn

But see http://scikit-learn.org/stable/install.html for more detailed instructions.

Then clone the repository to your local computer or one of our servers:

    git clone https://github.com/bricksdont/swiss-dialect-identification

## Usage

To split a train set into train and test sets, use

    python lala_team.py --split --data train.csv

To train a model, use

    python lala_team.py --train --model model_ngram_1_6_tfidf.pkl --data train.csv.split90 --verbose

To evaluate a trained model, use

    python lala_team.py --evaluate --samples train.csv.split10 --model model_ngram_1_6_tfidf.pkl --verbose

To use a trained model to make predictions for the test samples:

    python lala_team.py --predict --samples test.csv --model model_ngram_1_6_tfidf.pkl > Sandbox_tfidf_1_6.csv

For other options, use `--help`:

    python lala_team.py --help
