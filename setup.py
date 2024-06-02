from setuptools import setup, find_packages

setup(name='BCQ',
      version='1.0.0',
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=[
            "torch>=1.4.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4"
      ],
      )