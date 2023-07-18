# noinspection PyUnresolvedReferences
from logging import *
import sys


def configure(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=INFO):
    basicConfig(format=format, stream=stream, level=level)