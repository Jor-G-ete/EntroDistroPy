from setuptools import setup, find_packages

# variables to automatize the process
version = "0.6"
name_lib = "EntroDistroPy"

with open("README.md", "r") as rmd:
      long_description = rmd.read()

setup(
      name= name_lib,
      packages=find_packages(),
      version=version,
      license="GNU General Public License v3.0",
      description="Library which creates an entropy discretizier and overlays discretizers"
                  " from main libraries such a sikit-learn ",
      author="Jorge Lopez Marcos, Enrique F.Viamontes Pernas",
      author_email="jlomar2005@hotmail.com, envipe79@gmail.como",
      maintainer="Jorge Lopez Marcos",
      maintainer_email="jlomar2005@hotmail.com",
      url="https://github.com/Jor-G-ete/"+name_lib,
      download_url="https://github.com/Jor-G-ete/"+name_lib+"/archive/v"+version+".tar.gz",
      project_urls={
           "Documentation":"https://github.com/Jor-G-ete/"+name_lib,
           "Source Code":"https://github.com/Jor-G-ete/"+name_lib+"/blob/master/Entropy_dis.py"
      },
      platforms="Windows",
      keywords=["python3.7", "Maths", "scikit-learn", "Discretizier", "Entropy", "Binner"],
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=[
            'networkx',
            'numpy',
            'matplotlib',
            'pyyaml',
            'scikit-learn',
            'pandas',
            'scipy',
            'yellowbrick'
      ],
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',  # Define that your audience are developers
            'Topic :: Software Development :: Build Tools',
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
            'Programming Language :: Python :: 3.7',
            "Natural Language :: English",
            "Natural Language :: Spanish",
            "Operating System :: Microsoft :: Windows :: Windows 10",
            "Topic :: Scientific/Engineering :: Medical Science Apps."
            ],
      python_requires=">=3.7",
      )