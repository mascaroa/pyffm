from .pyffm import PyFFM

import sys
import logging

# TODO: propagate logging... tricky
log_fmt = f'[%(asctime)s.%(msecs)03d] - %(levelname)8s: %(name)10s -- %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_fmt)

print("        _________  _.\n.__    |__ |__ | \\/ | \n|__\\__/|   |   |    |\n|   /\n")
