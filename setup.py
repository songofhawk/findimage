import pathlib
from os import path
from setuptools import setup, find_packages


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
        name='findimage',
        version=parse_version('findimage'),
        description='to find a template image(smaller) in a source image(bigger)',
        long_description=parse_description(),
        long_description_content_type='text/markdown',
        url='https://github.com/songofhawk/findimage',
        author='songofhawk',
        author_email='songofhawk@gmail.com',
        license='MIT',
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent'
        ],
        include_package_data=True,
        install_requires=[
            'opencv_python>=3.0'
        ],
        packages=find_packages(include=['findimage', 'findimage.*']),
    )
