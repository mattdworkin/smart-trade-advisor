from setuptools import setup, find_packages

setup(
    name="smart-trade-advisor",
    version="0.1.0",
    description="An algorithmic trading recommendation system",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=2.3.2",
        "pandas>=2.0.0",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "scikit-learn>=1.2.2",
        "yfinance>=0.2.18",
        "websocket-client>=1.5.1",
        "requests>=2.30.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        'console_scripts': [
            'smart-trade=launch:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 