from abc import ABC, abstractmethod
from typing import Dict


class Loggable(ABC):
    """ Abstract class for objects that can be logged by the create_log_entry-endpoint """

    @abstractmethod
    def to_json(self) -> Dict:
        """ Returns a representation of the object as a dictionary """
        
        pass