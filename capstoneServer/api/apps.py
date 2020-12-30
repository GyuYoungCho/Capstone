from django.apps import AppConfig
import sys
from multiprocessing import Process
import multiprocessing
import threading

class ApiConfig(AppConfig):
    name = 'api'
    def ready(self):
        if 'runserver' in sys.argv:
            pass
        else:
            pass