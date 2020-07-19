import os

project_path = os.path.dirname(os.path.dirname(__file__))

us_tweets_path = project_path + "/data/us_trial.text"
us_labels_path = project_path + "/data/us_trial.labels"

es_tweets_path = project_path + "/data/es_trial.text"
es_labels_path = project_path + "/data/es_trial.text"



us_tweets_path_train = project_path + "/data/us_train.text"
us_labels_path_train = project_path + "/data/us_train.labels"
us_tweets_path_test = project_path + "/data/us_test.text"
us_labels_path_test = project_path + "/data/us_test.labels"
results_nb = project_path + "/results/nb.txt"
us_test_path = project_path + "/results/golden.txt"
