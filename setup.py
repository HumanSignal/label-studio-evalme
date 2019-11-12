import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='label-studio-evalme',
    version='0.0.1',
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
    python_requires='>=3.6',
)