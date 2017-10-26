A package for creating log entries for experiments. Accessible through the [```create_log_entry```](experiment_logger/experiment_logger.py#L133) function:


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
       """"
    
To integrate with other libraries, use the [```Loggable```](experiment_logger/loggable.py) "interface":

    class Loggable(ABC):
       """ Abstract class for objects that can be logged by the create_log_entry-endpoint """

       @abstractmethod
       def to_json(self) -> Dict:
           """ Returns a representation of the object as a dictionary """
        


