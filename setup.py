from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='baselines-pytorch',
      packages=[package for package in find_packages()
                if package.startswith('baselines')],
      install_requires=[
          'gym[mujoco,atari,classic_control]',
          'scipy',
          'tqdm',
          'joblib',
          'zmq',
          'dill',
          'torch >= 0.2.0',
          'azure==1.0.3',
          'progressbar2',
          'mpi4py',
          'pandas',
          'bokeh',
          'matplotlib',
          'seaborn',
      ],
      description="baselines-pytorch: implementations of reinforcement learning algorithms",
      author="Nadav Bhonkere",
      url='https://github.com/nadavbh12/baselines-pytorch',
      author_email="nadavbh@gmail.com",
      version="0.0.1")
