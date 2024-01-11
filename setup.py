from setuptools import setup

setup(
    name='themodel',
    version='1.0.0',
    description='A python package that takes your boring b&W manga and colorises it.',
    python_requires='>=3.10, <3.12',
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'pydantic',
        'pydantic_settings',
        'tqdm'
    ]
)
