from setuptools import setup, find_packages


def parse_requirements(filename):
    """Read requirements.txt and return list of dependencies."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    reqs = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return reqs


setup(
    name='feptp_pipeline',
    version='0.1.0',
    description='An open-source, end-to-end automated information extraction pipeline',
    packages=find_packages(),
    author='LinLe',
    include_package_data=True,
    package_data={
        'feptp_pipeline': [
            'resources/prompt/*.json',
            'resources/kb/*.csv',
        ]
    },
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.10',
)
