from parser_registry import ParserRegistry
import yolo_parser as YoloParser 


ParserRegistry.register("yolo", YoloParser)