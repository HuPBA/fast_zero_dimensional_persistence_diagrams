from distutils.core import setup, Extension
import numpy as np


def main():
    setup(name="fast_zero_persistence_diagram",
          version="1.0.0",
          description="A small package to compute persistence diagrams and persistence pairs for 0-dimensional Vietoris-Rips persistence.",
          author="Rubén Ballester Bautista",
          author_email="rballeba@gmail.com",
          ext_modules=[Extension("zero_persistence_diagram", ["single_linkage.c"],
                                 include_dirs=[np.get_include()])])


if __name__ == "__main__":
    main()
