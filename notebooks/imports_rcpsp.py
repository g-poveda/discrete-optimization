import logging

import nest_asyncio

# patch asyncio so that applications using async functions can run in jupyter
nest_asyncio.apply()
# set logging level
logging.basicConfig(level=logging.INFO)
