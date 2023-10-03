"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from usecase2 import __version__

PACKAGE_REQUIREMENTS = ["pyyaml"]

# packages for local development and unit testing
# please note that these packages are already available in DBR, there is no need to install them on DBR.
LOCAL_REQUIREMENTS = [
    "pyspark==3.2.1",
    "delta-spark==1.1.0",
    "mlflow",
    "boto3",
    "delta-spark==1.1.0",
    "scikit-learn==1.2.0",
    "databricks-sdk",
    "databricks-feature-store",
    #"databricks-registry-webhooks",
    "evidently",
    "pandas==1.5.3",
    "urllib3"
    # "xgboost<1.6"
    
]

TEST_REQUIREMENTS = [
    # development & testing tools
    "pytest",
    "coverage[toml]",
    "pytest-cov",
    "dbx>=0.8",
    "shap",
    "seaborn"
]

setup(
    name="usecase2",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["setuptools","wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"local": LOCAL_REQUIREMENTS, "test": TEST_REQUIREMENTS},
    entry_points = {
        "console_scripts": [
            "etl = usecase2.tasks.feature_eng_pipeline:entrypoint",
            "ml = usecase2.tasks.model_pipeline:entrypoint",
            "webhook = usecase2.tasks.webhook:entrypoint"
    ]},
    version=__version__,
    description="",
    author="",
)
