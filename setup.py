from setuptools import setup, find_packages

setup(
    name='voyage_agents',   
    packages=find_packages(include=['voyage_agents', 'voyage_agents.*', 'voyage_agents.core.*']),
)
