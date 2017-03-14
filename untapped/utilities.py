import datetime
import string
import os
import gzip
import pickle


def _repr(s):
    return repr(s).replace('\n','')


def toLog(prefix='',mydict={},**kwargs):
    if prefix != '':
        prefix += ' '
    logout = ''
    for k,v in mydict.items():
        logout += prefix+_repr(k)+': \t'+_repr(v)+'\n'
    for k,v in kwargs.items():
        logout += prefix+_repr(k)+': \t'+_repr(v)+'\n'
    return logout+'\n'


def toScreen(prefix='',mydict={},**kwargs):
    if prefix != '':
        prefix += ' '
    for k,v in mydict.items():
        print(prefix+_repr(k)+': \t'+_repr(v))
    for k,v in kwargs.items():
        print(prefix+_repr(k)+': \t'+_repr(v))
    print('')


def timeStamp(fmt='%Y-%m-%d-%H-%M-%S/{}'):
    return datetime.datetime.now().strftime(fmt)


def load_url(url,data_path):
    data_dir, data_file = os.path.split(data_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if data_dir == "" and not os.path.isfile(data_path):
        # Check if file is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            data_path
        )
        if os.path.isfile(new_path):
            data_path = new_path

    if (not os.path.isfile(data_path)):
        from six.moves import urllib
        origin = (url)
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, data_path)

    # Load the dataset
    with gzip.open(data_path, 'rb') as f:
        try:
            return pickle.load(f, encoding='latin1')
        except:
            return pickle.load(f)
