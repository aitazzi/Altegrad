import os
import csv


def _check_log_directory(directory):
    """
    Check that the log directory exists and create it if it doesn't.

    Args:
        directory: log directory path.

    Return:

    """

    try:
        if not os.path.exists(directory):
            print("Attempting to make log directory at " + directory)
            os.makedirs(directory)
    except IOError as e:
        sys.exit("Error attempting to create log directory: {0}".format(e.strerror).strip())


def _initialise_model_log(log_filepath):
    """Create a model logging file if it doesn't already exist.

    Args:
        log_filepath: log file path.

    Return:

    """

    if not os.path.exists(log_filepath):
        with open(log_filepath, 'a') as fp:
            a = csv.writer(fp, delimiter=',')
            data = [['DATETIME', 'Model', 'CV val scores', 'Mean CV val scores', 
            'CV train scores', 'Mean CV train scores', 
            'Parameters', 'Features importance']]
            a.writerows(data)