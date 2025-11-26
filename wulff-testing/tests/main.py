import os
import sys
import unittest
import doctest


class ScriptTestCase(unittest.TestCase):
    def __init__(self, methodname='testfile', filename=None):
        unittest.TestCase.__init__(self, methodname)
        self.filename = filename

    def testfile(self):
        try:
            with open(self.filename) as fd:
                exec(compile(fd.read(), self.filename, 'exec'), {})
        except KeyboardInterrupt:
            raise RuntimeError('Keyboard interrupt')
        except ImportError as ex:
            module = ex.args[0].split()[-1].replace("'", '').split('.')[0]
            if module in ['matplotlib']:
                raise unittest.SkipTest('no {} module'.format(module))
            else:
                raise

    @property
    def id(self):
        return self.filename.split('tests')[-1]

    def __str__(self):
        return self.id

    def __repr__(self):
        return f'ScriptTestCase(filename="{self.filename}")'


def find_potential_doctest_files(suite, script_dir):
    tests = []
    for root, dirs, files in os.walk(script_dir):
        for f in files:
            if f.endswith('.py'):
                tests.append(os.path.join(root, f))
    tests.sort()
    suite.addTests(doctest.DocFileSuite(*tests, module_relative=False))


if __name__ == '__main__':

    # Find testing dirs
    main_dir = os.path.dirname(__file__)
    unittest_dir = os.path.join(main_dir, 'unittest')
    module_dir = os.path.join(main_dir, '../wulffpack')

    # Collect tests
    suite = unittest.TestLoader().discover(unittest_dir, pattern='*.py')
    find_potential_doctest_files(suite, module_dir)

    # Run tests
    ttr = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    results = ttr.run(suite)

    assert len(results.failures) == 0, 'At least one test failed'
    assert len(results.errors) == 0, 'At least one test failed'
