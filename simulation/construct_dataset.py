import json
import random
import os
import subprocess
import tempfile
import json

template_file_prefix = './'

with open('curriculum.json', 'r') as f:
    data = json.load(f)

processes = []

for dataset in ['train', 'test', 'val']:
    count_downscale = data[dataset + '_count_downscale']

    for c, category in enumerate(data['categories']):
        f = tempfile.TemporaryFile()
        dir_name = 'tmp_' + dataset + "-" + str(c)
        num_examples = category['count'] // count_downscale
        command = f'python construct.py --template_file {os.path.join(template_file_prefix, category["template"])} --dataset_dir {dir_name} --num_examples {num_examples} --use_panda {data["use_panda"]}'
        p = subprocess.Popen(command.split(), stdout=f)
        processes.append((p,f))

logfile = open('log.txt', 'wb')

for p, f in processes:
    p.wait()
    f.seek(0)
    logfile.write(f.read())
    f.close()
