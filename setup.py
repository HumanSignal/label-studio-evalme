import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

# Module dependencies
requirements, dependency_links = [], []
with open('requirements.txt') as f:
    for line in f.read().splitlines():
        if line.startswith('-e git+') or line.startswith('http'):
            dependency_links.append(line.replace('-e ', ''))
        else:
            requirements.append(line)

setuptools.setup(
    name='label-studio-evalme',
    version='0.0.17',
    author='Heartex',
    author_email="hello@heartex.ai",
    description='Evaluation metrics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/heartexlabs/label-studio-evalme',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
	dependency_links=dependency_links,
    python_requires='>=3.6',
)
