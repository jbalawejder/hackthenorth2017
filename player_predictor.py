from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import tensorflow as tf


BASKETBALL_TRAINING_COLUMNS = [
    "age", "height", "weight", "ppg", "fgp", "assists", "rebounds", "steals", "blocks", "3ptp", "position",
]

BASKETBALL_EVALUATION_COLUMNS = [
	"age", "height", "weight", "ppg", "fgp", "assists", "rebounds", "steals", "blocks", "3ptp", "position",
]
 

# Continuous base columns.
age 	 = tf.feature_column.numeric_column("age")
height 	 = tf.feature_column.numeric_column("height")
weight 	 = tf.feature_column.numeric_column("weight")
ppg 	 = tf.feature_column.numeric_column("ppg") # points per game
fgp 	 = tf.feature_column.numeric_column("fgp") # field goal percentage
assists  = tf.feature_column.numeric_column("assists")
rebounds = tf.feature_column.numeric_column("rebounds")
steals 	 = tf.feature_column.numeric_column("steals")
blocks 	 = tf.feature_column.numeric_column("blocks")
3ptp 	 = tf.feature_column.numeric_column("3ptp") # 3pt percentage

position = tf.feature_column.categorical_column_with_vocabulary_list(
    "position", ["PG", "SG", "SF", "PF", "C"])


# Wide columns and deep columns.
base_columns = [
	age, height, weight, ppg, fgp, assists, rebounds, steals, blocks, 3ptp, position,
]

#feature crossings
crossed_columns = [
    tf.feature_column.crossed_column(
        ["height", "blocks"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, "height", "rebounds"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ["age", "ppg"], hash_bucket_size=1000)
    tf.feature_column.crossed_column(
        ["age", "ppg"], hash_bucket_size=1000)
    tf.feature_column.crossed_column(
        ["position", "height"], hash_bucket_size=1000)
    tf.feature_column.crossed_column(
        ["position", "fpg"], hash_bucket_size=1000)
]

deep_columns = [
    tf.feature_column.indicator_column(position),
    age, 
    height, 
    weight, 
    ppg, 
    fgp, 
    assists,
    rebounds, 
    steals, 
    blocks, 
    3ptp,
    position,
]


def maybe_download(train_data, test_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s"% test_file_name)

  return train_file_name, test_file_name


def build_estimator(model_dir, model_type):
  """Build an estimator."""
  if model_type == "wide":
    m = tf.estimator.LinearClassifier(
        model_dir=model_dir, feature_columns=base_columns + crossed_columns)
  elif model_type == "deep":
    m = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=[100, 50])
  else:
    m = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=crossed_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m


def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  train_file_name, test_file_name = maybe_download(train_data, test_data)
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir

  m = build_estimator(model_dir, model_type)
  # set num_epochs to None to get infinite stream of data.
  m.train(
      input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
      steps=train_steps)
  # set steps to None to run evaluation until all data consumed.
  results = m.evaluate(
      input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
      steps=None)
  print("model directory = %s" % model_dir)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=2000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

