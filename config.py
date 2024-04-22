import os
import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.SafeLoader)
            # self._dict['PATH'] = os.getcwd() # current directory
            
    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        # if DEFAULT_CONFIG.get(name) is not None:
        #     return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')