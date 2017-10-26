import pytest
import numpy as np

from experiment_logger.experiment_logger import create_experiment_summary, _calculate_accuracy, _compute_confusion_matrix, _extract_errors


def test_accuracy():
    true = np.asarray([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0]
    ])

    preds = np.asarray([
        [0.6, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.7],
        [0.4, 0.0, 0.5, 0.1]
    ])

    acc = _calculate_accuracy(true, preds)

    assert abs(0.6 - acc) < 1e-5, '_calculate_accuracy computes accuracy wrong'


def test_confusion_matrix():
    true = np.asarray([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0]
    ])

    preds = np.asarray([
        [0.6, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.7],
        [0.4, 0.0, 0.5, 0.1]
    ])

    conf_matrix = _compute_confusion_matrix(true, preds)

    expected_conf_matrix = np.asarray([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    assert np.array_equal(expected_conf_matrix, conf_matrix), '_compute_confusion_matrix computes confusion matrix wrong'


def test_experiment_summary_without_errors():
    true = np.asarray([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0]
    ])

    preds = np.asarray([
        [0.6, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.7],
        [0.4, 0.0, 0.5, 0.1]
    ])

    expected_conf_matrix = np.asarray([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    ret = create_experiment_summary(true, preds)

    assert 2 == len(ret), 'create_experiment_summary with no paths does not return (acc, confusion_matrix)'
    assert type(ret[0]) is np.float64, 'create_experiment_summary with no paths does not return (acc, confusion_matrix)'
    assert type(ret[1]) is np.ndarray, 'create_experiment_summary with no paths does not return (acc, confusion_matrix)'


def test_extract_errors():
    true = np.asarray([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0]
    ])

    preds = np.asarray([
        [0.6, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.7],
        [0.4, 0.0, 0.5, 0.1]
    ])

    paths = np.asarray(['path1', 'path2', 'path3', 'path4', 'path5'])

    errors = _extract_errors(true, preds, paths)

    erronous_paths = {
        'path1': {'label': 1, 'predicted': 0, 'confidences': [0.6, 0.2, 0.1, 0.1]},
        'path5': {'label': 0, 'predicted': 2, 'confidences': [0.4, 0.0, 0.5, 0.1]}
    }

    assert 2 == len(errors), '_extract_errors returns the wrong number of errors'

    for error in errors:
        path = error['path']
        assert path in erronous_paths, '_extract_errors extracts a false error'
        assert error['label'] == erronous_paths[path]['label'], '_extract_errors returns the wrong label for an error'
        assert error['predicted'] == erronous_paths[path]['predicted'], '_extract_errors returns the wrong label for an error'
        assert np.array_equal(error['confidences'], erronous_paths[path]['confidences']), '_extract_errors returns the wrong confidences for an error'


def test_experiment_summary():
    true = np.asarray([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0]
    ])

    preds = np.asarray([
        [0.6, 0.2, 0.1, 0.1],
        [0.0, 1.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.7],
        [0.4, 0.0, 0.5, 0.1]
    ])

    paths = np.asarray(['path1', 'path2', 'path3', 'path4', 'path5'])

    ret = create_experiment_summary(true, preds, paths)

    assert 3 == len(ret), 'create_experiment_summary with paths does not return (acc, confusion_matrix, errors)'
    assert type(ret[0]) is np.float64, 'create_experiment_summary with paths does not return (acc, confusion_matrix, errors)'
    assert type(ret[1]) is np.ndarray, 'create_experiment_summary with paths does not return (acc, confusion_matrix, errors)'
    assert type(ret[2]) is list, 'create_experiment_summary with paths does not return (acc, confusion_matrix, errors)'

