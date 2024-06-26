from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, Tuple

class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        if name in self._obj_map:
            raise ValueError('An object named \'{}\' was already registered in \'{}\' registry!'.format(
                name, self._name
            ))

        self._obj_map[name] = obj

    def register(self, obj: Any = None, name: str = None) -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """

        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                self._do_register(func_or_class.__name__ if name is None else name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        self._do_register(obj.__name__ if name is None else name, obj)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                'No object named \'{}\' found in \'{}\' registry!'.format(name, self._name)
            )
        return ret

    def build(self, name: str, params: Dict[str, Any] = None):
        if params is None:
            params = {}
        else:
            params = deepcopy(params)
        return self.get(name)(**params)

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        return 'Registry of {}:\n{}'.format(self._name, self._obj_map)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__

SCALER_REGISTRY = Registry("Scaler")
