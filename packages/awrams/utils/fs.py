import os
import re
import fnmatch

def apply_prefix(filemap,prefix):
    '''
    Apply a common prefix (such as a base path) each value in a dictionary, where
    the values represent a filename / relative path.

    Returns new dictionary with same keys as filemap
    '''
    return {k:(prefix+v) for k,v in filemap.items()}

def recursive_glob(path, pattern):
    '''
    to glob a path with wildcards pass arg "path" as glob.glob('/path/*/with/*/wildcards/')
    '''
    
    def find_files(path, pattern):
        fs = [os.path.join(_path, filename)
              for _path, dirnames, filenames in os.walk(path)
              for filename in filenames
              if fnmatch.fnmatch(filename, pattern)]
        return fs

    fs = []
    if type(path) == list:
        for _path in path:
            fs += find_files(_path, pattern)
    else:
        fs = find_files(path, pattern)
    return fs

def directory_empty(path):
    """
    Returns true if the directory is empty. False otherwise
    """
    return len(os.listdir(path))==0

class FileMatcher(object):
    def __init__(self,search_pattern,extraction_pattern=None):
        self.file_search_pattern = search_pattern
        
        # +++ Unused!
        if extraction_pattern is None:
            extraction_pattern = r"(?P<year>[1-2][0-9][0-9][0-9])"
        self.extraction_pattern = re.compile(extraction_pattern)

    def locate(self):

        directory = os.path.dirname(self.file_search_pattern)
        if not directory: directory = '.'
        reg_exp = re.compile(fnmatch.translate(os.path.basename(self.file_search_pattern)),re.IGNORECASE)
        return [os.path.join(directory,fn) for fn in os.listdir(directory) if reg_exp.match(fn)]

def md5_file(fn):
    '''
    Compute an MD5 hashsum of the binary contents of a file.
    '''
    # from www.pythoncentral.io/hashing-files-with-python
    import hashlib
    BLOCKSIZE=64*1024*1024
    hasher = hashlib.md5()
    with open(fn,'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

