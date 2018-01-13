"""Test data set."""
import os
import glovar


DEFAULT_FILE_PATH = os.path.join(glovar.DATA_DIR, 'testdata', 'testdata.csv')


def generate_testdata(target=DEFAULT_FILE_PATH):
    """Generates the test data."""
    print('Generating test data..')
    data = [
        ('B', 'A'),
        ('C', 'A'),
        ('D', 'A'),
        ('E', 'B'),
        ('F', 'B'),
        ('G', 'C'),
        ('H', 'C'),
        ('I', 'C'),
        ('J', 'D')]
    print('Writing file...')
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'w') as file:
        file.write('child,parent\n')
        for u, v in data:
            file.write('%s,%s\n' % (u, v))
    print('Success.')
