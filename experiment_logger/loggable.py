import json

from abc import ABC
from collections import OrderedDict
from importlib import import_module


class ObjectDict(OrderedDict):

    def __init__(self, *args, obj_cls: type = None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            if obj_cls:
                self._obj_cls = obj_cls
                self['_obj_cls'] = {'module_name': obj_cls.__module__, 'name': obj_cls.__qualname__}
            else:
                object_cls_module = import_module(self['_obj_cls']['module_name'])
                self._obj_cls = getattr(object_cls_module, self['_obj_cls']['name'])
            assert issubclass(self._obj_cls, Loggable)
        except (KeyError, AssertionError):
            raise ValueError('ObjectDict must be initialised with a Loggable type or a dict with an _obj_cls key')

    def to_object(self):
        return self._obj_cls.from_object_dict(self)


class Loggable(ABC):
    """ Abstract class for objects that can be logged by the create_log_entry-endpoint """

    def to_object_dict(self, **kwargs) -> OrderedDict:
        """ Returns a representation of the object as a dictionary """

        dct = ObjectDict({}, obj_cls=type(self))

        for key, attribute in kwargs.items():
            if issubclass(type(attribute), Loggable):
                dct[key] = attribute.to_object_dict()
            else:
                dct[key] = attribute
                # TODO: handle numpy arrays

        return dct

    @classmethod
    def from_json_string(cls, strng):
        try:
            dct = json.loads(strng)
        except json.JSONDecodeError:
            raise ValueError('strng must be valid json string')
        return cls.from_object_dict(dct)

    @classmethod
    def from_object_dict(cls, dct, **kwargs):
        """ Recreates object from dictionary """
        obj = cls.__new__(cls)
        for key, value in kwargs.items():
            try:
                obj_dct = ObjectDict(value)
            except (TypeError, ValueError):
                setattr(obj, key, value)
            else:
                setattr(obj, key, obj_dct.to_object())

        return obj
