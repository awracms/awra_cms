import os
import importlib.machinery
import types
from pathlib import Path


def _mod_from_file(packagename, filename):
    loader = importlib.machinery.SourceFileLoader(packagename, str(filename))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


def get_awrams_base_path():
    base_path = os.environ.get('AWRAMS_BASE_PATH')
    if base_path is None:
        raise Exception("AWRAMS_BASE_PATH environment variable has not been set")

    if os.name is 'nt':
        if '"' in base_path:
            # Windows environment variable is coming in with a double quote instead of
            # the backslash. Since double quote is not valid in windows file paths
            # (source: https://docs.microsoft.com/en-us/windows/desktop/msi/filename)
            # just replace it.
            base_path = base_path.replace('"', '\\')
        if base_path.startswith('\\'):
            # When running in CI on windows we get backslashes at the start of the path
            # which makes it not a valid path
            base_path = base_path[1:]
    return Path(base_path)

def get_awrams_data_path():
    data_path = os.environ.get('AWRAMS_DATA_PATH')
    if data_path is None:
        return get_awrams_base_path() / 'data'

    if os.name is 'nt':
        if '"' in data_path:
            # Windows environment variable is coming in with a double quote instead of
            # the backslash. Since double quote is not valid in windows file paths
            # (source: https://docs.microsoft.com/en-us/windows/desktop/msi/filename)
            # just replace it.
            data_path = data_path.replace('"', '\\')
        if data_path.startswith('\\'):
            # When running in CI on windows we get backslashes at the start of the path
            # which makes it not a valid path
            data_path = data_path[1:]
    return Path(data_path)

def _get_default_system_profile():
    profile = os.environ.get('AWRAMS_DEFAULT_SYSTEM_PROFILE')
    if profile is None:
        profile = 'default'
    return profile

def get_config_path(subpath='',remote=False):
    if remote:
        base_path = get_system_profile().get_settings()['DATA_PATHS']['AWRAMS_BASE']
    else:
        base_path = get_awrams_base_path()
    return Path(base_path) / 'config' / subpath

def get_system_profile(profile=None):
    if profile is None:
        profile = os.environ.get('AWRAMS_SYSTEM_PROFILE')
    if profile is None:
        profile = _get_default_system_profile()

    profile_path = get_awrams_base_path() / 'config' / 'system' / ('%s.py' % profile)
    return _mod_from_file('awrams_system_config', profile_path)


def get_model_profile(model, profile):
    profile_path = get_awrams_base_path() / 'config' / 'models' / model / ('%s.py' % profile)
    return _mod_from_file('awrams_model_config', profile_path)

def get_profile(profile):
    base_path = get_awrams_base_path()
    profile_path = os.path.join(base_path,'config','%s.py' % profile)
    return _mod_from_file('awrams_profile', profile_path)

def set_active_system_profile(profile=None):
    if profile is None:
        if 'AWRAMS_SYSTEM_PROFILE' in os.environ:
            os.environ.pop('AWRAMS_SYSTEM_PROFILE')
    else:
        os.environ['AWRAMS_SYSTEM_PROFILE'] = profile

def set_awrams_base_path(path):
    os.environ['AWRAMS_BASE_PATH'] = path

def set_awrams_data_path(path):
    os.environ['AWRAMS_DATA_PATH'] = path
