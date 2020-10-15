import json
import os


def write_to_json(data):
    file_name = os.environ["expt_name"]
    file_path = os.environ["file_path"]
    with open(file_path + '/' + file_name + '.json', "r+") as fp:
        # https://stackoverflow.com/questions/47792142/how-to-check-if-json-file-contains-only-empty-array
        if os.path.getsize(file_path + '/' + file_name + '.json') > 2:
            log = json.load(fp)
            log.update(data)
            fp.seek(0)
            json.dump(log, fp)
        else:
            json.dump(data, fp)
