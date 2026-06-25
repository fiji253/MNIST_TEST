#class ParserRegistry: 
 #   parsers = {} 
  #  @classmethod 
   # def register(cls, name, parser_class): 
    #    cls.parsers[name] = parser_class 
    #@classmethod 
    #def create(cls, name): 
     #   return cls.parsers[name]()
from abc import ABC, abstractmethod
from core_format import AnnotationSample

class AbstractParser(ABC):
    @abstractmethod
    def parse(self) -> list[AnnotationSample]:
        pass

class ParserRegistry:
    parsers = {}

    @classmethod
    def register(cls, name: str, parser_class: type[AbstractParser]):
        if name in cls.parsers:
            raise ValueError(f"Parser already registered: {name}")
        
        cls.parsers[name] = parser_class
    
    @classmethod
    def create(cls, name: str, *args, **kwargs):   # name this is data format
        if name not in cls.parsers:
            raise ValueError(f"parser: {name} - not found")
        parser_class = cls.parsers[name]
        return parser_class(*args, **kwargs)
