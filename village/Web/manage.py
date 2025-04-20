#!/usr/bin/env python
import os
import sys
from Config.get_config import load_config
import warnings
import Config.pytorch as pytorch_model
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    config = load_config()
    Web_Port = str(config['Web_Port'])
    Djnago_Pram = pytorch_model.get_django_pram()

    os.environ.setdefault(Djnago_Pram['django_key'], Djnago_Pram['django_value'])
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    # Add default arguments for runserver
    if len(sys.argv) == 1:
        sys.argv += [Djnago_Pram['django_run'], "127.0.0.1:" + Web_Port]

    execute_from_command_line(sys.argv)
