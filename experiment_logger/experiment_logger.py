import json
import os
import time
import numpy as np
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

from .loggable import Loggable


def _sort_keys(keys: List[str]) -> List[str]:
    """ Keys are sorted in the following order (for readability of the logs):

    name
    timestamp
    project
    accuracy
    confusion_matrix
    errors
    model
    model_description
    preprocessor
    ** EVERYTHING ELSE IN RANDOM ORDER **
    dataset_description
    trainset_description
    testset_description
    dataset
    trainset
    testset  

    If any of the keys are missing they will simply be dropped
    """

    sorted_keys = []
    if 'name' in keys:
        sorted_keys.append('name')
    if 'timestamp' in keys:
        sorted_keys.append('timestamp')
    if 'project' in keys:
        sorted_keys.append('project')
    if 'accuracy' in keys:
        sorted_keys.append('accuracy')

    # Only allows for either confusion_matrix or conf_matrix
    if 'confusion_matrix' in keys:
        sorted_keys.append('confusion_matrix')
    elif 'conf_matrix' in keys:
        sorted_keys.append('conf_matrix')

    if 'errors' in keys:
        sorted_keys.append('errors')
    if 'model' in keys:
        sorted_keys.append('model')
    if 'model_description' in keys:
        sorted_keys.append('model_description')
    if 'preprocessor' in keys:
        sorted_keys.append('preprocessor')

    if 'dataset_description' in keys:
        sorted_keys.append('dataset_description')
    if 'trainset_description' in keys:
        sorted_keys.append('trainset_description')
    if 'testset_description' in keys:
        sorted_keys.append('testset_description')
    if 'dataset' in keys:
        sorted_keys.append('dataset')
    if 'trainset' in keys:
        sorted_keys.append('trainset')
    if 'testset' in keys:
        sorted_keys.append('testset')

    return sorted_keys


def _serialize_list(l: List[Any]) -> List[Any]:
    """ Recursively make a list json-serializable """
    for i in range(len(l)):
        if type(l[i]) is list:
            l[i] = _serialize_list(l[i])
        elif type(l[i]) is dict:
            l[i] = _serialize_collection(l[i])

        try:
            json.dumps(l[i])
        except Exception:
            if issubclass(type(l[i]), Loggable):
                json.dumps(l[i].to_json)
            else:
                l[i] = str(l[i])

    return l


def _serialize_collection(collection: Dict[Any, Any]) -> Dict[Any, Any]:
    """ Recursively makes both keys and values of a dictionary json-serializable """

    keys_to_drop = []
    for key in collection:
        try:
            json.dumps(key)
        except Exception:
            try:
                collection[str(key)] = collection[key]
                del collection[key]
                key = str(key)
            except Exception:
                print('Unable to serialize keyword %s. Dropped from collection' % key)

        if type(collection[key]) is dict:
            collection[key] = _serialize_collection(collection[key])
        elif type(collection[key]) is list:
            collection[key] = _serialize_list(collection[key])

        try:
            json.dumps(collection[key])
        except Exception:
            try:
                if issubclass(type(collection[key]), Loggable):
                    collection[key] = collection[key].to_json()
                else:
                    collection[key] = str(collection[key])
            except Exception as e:
                print('Unable to serialize value for keyword %s with type %s. Dropped from collection' % (key, type(collection[key]).__name__))
                keys_to_drop.append(key)

    for key in keys_to_drop:
        del collection[key]

    return collection


def create_log_entry(name: str, *, logdir: str = None, y: np.ndarray = None, preds: np.ndarray = None, paths: Union[List[str], np.ndarray] = None, **kwargs: Dict[str, Any]):
    """ Creates a log file which is meant to represent an experiment. Usually just dumps string encoded versions of the
    key/value pairs to a json, but some combinations of parameters are treated specially (see below)

    Keyword arguments:
    -----------------
    name : str
        The name of the experiment. Also used to name the log file

    logdir : str (optional)
        The folder where the logfile should be stored

    y : np.ndarray (optional)
        The ground truth labels for the dataset used in the experiment

    preds : np.ndarray (optional)
        The predicted class probabilities for the datapoints used in the experiment

    paths : Union[List, np.ndarray] (optional)
        Paths for the images used as datapoints (if applicable)

    Combinations:
    ------------
    y and preds
        An experiment summary is created

    y, preds and paths
        An experiment summary with errors is created

    y, preds and a testset which has a paths property
        An experiment summary with errors is created

    Returns:
    -------
    The filename of the logfile
    """
    
    filepath = '%s_%d.log' % (name, int(time.time()))

    if logdir is not None:
        filepath = os.path.join(logdir, filepath)

    # If no paths is given, but a testset with paths is given, extract paths from this
    if paths is None and 'testset' in kwargs and hasattr(kwargs['testset'], 'paths'):
        paths = kwargs['testset'].paths

    # If we have results, create an experiment summary
    if y is not None and preds is not None:
        # If we also have paths, create an experiment summary with error descriptions
        if paths is not None:
            acc, conf_matrix, errors = create_experiment_summary(y, preds, paths=paths)
            kwargs['errors'] = errors
        else: 
            acc, conf_matrix = create_experiment_summary(y, preds)

        kwargs['accuracy'] = acc
        kwargs['conf_matrix'] = conf_matrix

    kwargs['name'] = name
    kwargs['timestamp'] = datetime.now()
    

    collection = _serialize_collection(kwargs)
    sort_order = _sort_keys([key for key in collection])
    ordered_collection = OrderedDict()

    for key in sort_order:
        ordered_collection[key] = collection[key]

    with open(filepath, 'w') as f:
        json.dump(ordered_collection, f, indent=4)

    return filepath


def _extract_errors(true: np.ndarray, preds: np.ndarray, paths: Union[List[str], np.ndarray]) -> List:
    """ Creates a set of error objects on the defined format (see below) from a set of ground truth labels and predictions.
    
    Keyword arguments:
    -----------------
    true : np.ndarray
        The onehot encoded ground truth labels

    preds : np.ndarray
        The predicted class probabilities
    
    Returns:
    -------
    List[
        {
            \'path\': str,
            \'label\': int,
            \'predicted\': int,
            \'confidences\': np.ndarray 
        }
    ]
    """

    errors = []

    for i in range(len(true)):
        if np.argmax(true[i]) != np.argmax(preds[i]):
            error = {}
            error['path'] = paths[i]
            error['label'] = np.argmax(true[i])
            error['predicted'] = np.argmax(preds[i])
            error['confidences'] = preds[i]
            errors.append(error)


    return errors


def _calculate_accuracy(true: np.ndarray, preds: np.ndarray) -> np.float64:
    """ Calculates the accuracy between a set of ground truth labels and predictions.

    WARNING: Subject to (very small) rounding errors

    Keyword arguments:
    -----------------
    true : np.ndarray
        The onehot encoded ground truth labels

    preds : np.ndarray
        The predicted class probabilities
    """

    argmaxed_true = np.argmax(true, axis=1)
    argmaxed_preds = np.argmax(preds, axis=1)

    equal = np.equal(argmaxed_true, argmaxed_preds)

    return np.sum(equal) / len(equal)


def _compute_confusion_matrix(true: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """ Computes the confusion matrix between a set of ground truth labels and predictions.

    Keyword arguments:
    -----------------
    true : np.ndarray
        The onehot encoded ground truth labels

    preds : np.ndarray
        The predicted class probabilities
    """

    _, num_classes = true.shape
    conf_matrix = np.zeros((num_classes, num_classes))

    argmaxed_true = np.argmax(true, axis=1)
    argmaxed_preds = np.argmax(preds, axis=1)

    for i in range(len(argmaxed_true)):
        conf_matrix[argmaxed_true[i]][argmaxed_preds[i]] += 1

    return conf_matrix.astype(int)


def create_experiment_summary(true: np.ndarray, preds: np.ndarray, paths: Union[List[str], np.ndarray] = None) -> Union[Tuple[np.float64, np.ndarray], Tuple[np.float64, np.ndarray, List]]:
    """ Creates an experiment summary from a set of ground truth labels and predictions. If labels
    are also provided, the actual errors are also returned.

    Keyword arguments:
    -----------------
    true : np.ndarray
        The onehot encoded ground truth labels

    preds : np.ndarray
        The predicted class probabilities

    paths : List[str] (optional)
        A list of paths to images, which can be used to generate error objects
    """
    acc = _calculate_accuracy(true, preds)
    conf_matrix = _compute_confusion_matrix(true, preds)

    if paths is not None:
        errors = _extract_errors(true, preds, paths)
        return acc, conf_matrix, errors

    return acc, conf_matrix
