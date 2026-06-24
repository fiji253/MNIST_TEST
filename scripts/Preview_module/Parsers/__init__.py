from parser_registry import ParserRegistry
from .yolo_parser import YOLO_Parser


ParserRegistry.register("yolo", YOLO_Parser)