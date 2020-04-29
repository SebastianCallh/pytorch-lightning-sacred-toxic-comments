from setuptools import setup

setup(
   name='toxic-comments',
   version='0.1',
   description='A scaffolding for data science projects.',
   author='Sebastian Callh',
   author_email='sebastian.callh@peltarion.com',
   packages=[
       'data',
       'model',
       'tracking'
   ],
   install_requires=[
       'torch >= 1.4, < 2.0'
   ]
)
