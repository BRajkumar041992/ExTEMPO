from setuptools import setup, find_packages

setup(
    name="extempo",
    version="2.0",
    description="ExTEMPO: A tool for simulating solar-like oscillations and power spectra",
    author="Brandon Rajkumar",
    author_email="Brandon.Rajkumar@warwick.ac.uk",
    url="https://github.com/BRajkumar041992/ExTEMPO",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.7",
)
