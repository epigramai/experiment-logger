import sys

def pytest_sessionstart(session):
    sys.path.append('.')
    print('SYS:PATH:AFTER: %s' % str(sys.path))