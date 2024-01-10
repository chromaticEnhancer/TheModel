from setuptools import setup

setup(
    name='themodel',
    version='1.0.0',
    description='A python package that takes your boring b&W manga and colorises it.',
    python_requires='>=3.10, <3.12',
    install_requires=[
        'torch==2.1.2+cu121',
        'torchvision==0.16.2+cu121',
        'matplotlib',
        'pydantic',
        'pydantic_settings',
        'tqdm'
    ]
)
