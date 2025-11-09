from setuptools import setup, find_packages

setup(
    name="trading_assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "requests",
        "streamlit",
        "schedule",
        "plotly"
    ],
    python_requires='>=3.7',
)
