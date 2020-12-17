from .pyctr import PyCTR

import sys
import logging

# TODO: propagate logging... tricky
log_fmt = f'[%(asctime)s.%(msecs)03d] - %(levelname)8s: %(name)10s -- %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_fmt)
