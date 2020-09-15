import os
import urllib.request

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
from d4rl.infos import DATASET_URLS  # noqa

# >>>> Write the destination directory here
dest = None
# <<<< Write the destination directory here
if dest is None:
    raise ValueError("you need to manually fill in the destination directory.")

for env_id, url in DATASET_URLS.items():
    print("{}. Downloading from url: {}".format(env_id, url))
    urllib.request.urlretrieve(url, os.path.join(dest, env_id) + '.h5')
