import datetime
import string


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
