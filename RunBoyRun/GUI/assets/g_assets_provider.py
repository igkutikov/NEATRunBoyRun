import typing
import os

class AssetsProvider:
    def __new__(cls):
        def __deleted_new(cls_):
            return cls_.__instances
        cls.__instances =  object.__new__(cls)
        cls.__new__ = __deleted_new
        return cls.__instances


    def __init__(self):
        def __deleted_init__(self2: AssetsProvider):
            return
        self.__deleted_init__ = __deleted_init__

        self.__resources_dir = os.path.dirname(os.path.abspath(__file__))
        self.__resources: typing.Dict[str, str] = {
            "arrow": os.path.join(self.__resources_dir, "arrow.png"),
        }


    def get_resource(self, resource_name: str) -> str:
        resource: typing.Optional[str] = self.__resources.get(resource_name, None)
        if resource is None:
            raise KeyError(f"Resource '{resource_name}' not found.")
        return resource