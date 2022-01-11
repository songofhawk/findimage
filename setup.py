import pathlib
from os import path
from setuptools import setup, find_packages

# for pip >= 10
try:
    from pip._internal.req import parse_requirements
# for pip <= 9.0.3
except ImportError:
    from pip.req import parse_requirements

PACKAGE_ROOT = pathlib.Path(__file__).parent


def parse_version(package):

    init_file = f'{PACKAGE_ROOT}/{package}/__init__.py'
    with open(init_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if '__version__' in line:
                return line.split('=')[1].strip()[1:-1]
    return ''


def parse_description():
    """
    Parse the description in the README file
    """
    readme_file = f'{PACKAGE_ROOT}/README.md'
    if path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            long_description = f.read()
        return long_description
    return ''


if __name__ == '__main__':

    setup(
        name='xone',
        version=parse_version('xone'),
        description='Frequently used functions for financial data analysis',
        long_description=parse_description(),
        long_description_content_type='text/markdown',
        url='https://github.com/alpha-xone/xone',
        author='Alpha x1',
        author_email='alpha.xone@outlook.com',
        license='Apache',
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        include_package_data=True,
        install_requires=[
            str(getattr(ir, 'req' if hasattr(ir, 'req') else 'requirement'))
            for ir in parse_requirements(
                f'{PACKAGE_ROOT}/requirements.txt', session='hack'
            )
        ],
        packages=find_packages(include=['xone', 'xone.*']),
    )
