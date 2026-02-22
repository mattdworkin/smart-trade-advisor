from setuptools import setup, find_packages

setup(
    name="smart-trade-advisor",
    version="1.0.0",
    description="Role-aware market research agent with vector and graph retrieval",
    author="Smart Trade Advisor",
    author_email="support@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.8.0",
        "pydantic-settings>=2.4.0",
        "requests>=2.32.0",
        "feedparser>=6.0.11",
        "beautifulsoup4>=4.12.3",
        "yfinance>=0.2.54",
        "apscheduler>=3.10.4",
        "psycopg[binary]>=3.2.1",
        "pgvector>=0.3.6",
        "neo4j>=5.24.0",
        "openai>=1.40.0",
    ],
    entry_points={
        'console_scripts': [
            'smart-trade-agent=smart_trade_agent.cli:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 
