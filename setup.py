from setuptools import find_packages,setup 
from typing import List 


def get_requirements(file_path:str)->List[str]:

    requirement_list:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement!= "-e .":
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("File is not found")

    return requirement_list 

setup(
    name= 'PneumoScan-AI',
    version= '1.0',
    author= 'Ahmed',
    author_email= 'tanvirahmed754575@gmail.com',
    packages= find_packages(),
    install_requires= get_requirements(file_path='requirements.txt')
)