import os
import json

def save_arguments(output_path, args):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'arguments.json'), 'w') as file:
        json.dump(args.__dict__, file, indent=4, sort_keys=True)
