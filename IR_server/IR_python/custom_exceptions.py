# -*- coding: utf-8 -*-

class NoIndexFilesException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
class IniException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
