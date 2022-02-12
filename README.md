multiwell_project
==============================

# WellPlate Code Project

Code Project for the [multiwell project] in FRA (http://.it/).



# Code Project
The project is named 

This code represents


This project has various specific `README` in the project structure.
A  `README` in the main source code folder `multiwell_project`, describing the overall folders and portions of the pipeline.
Each of the 5 python modules is described in this read me.
A  `README` in `data` folder ideally describes the content of what soul be present in that folder. No data is provided, given the big volume of such data, and privacy concerns.


## Code Development
As data scientists and teams can’t rely on the tools and processes that software engineering teams have been using for the last 20 years for machine learning (ML), this code has to be intended developed with 1-day sprints, so code is modular, and after a year-long development all these pieces are gathered together in this pipeline project.

## Code structure
The initial Data Science pipeline gets its idea from IBM's and the structure is also extended by Cookiecutter example and more referenced best practical tutorials.
### Data ETL
This step involves:
* accessing the data source
* basic exploration of the data source, for a first primary idea
* transforming the data, so it can be easily worked with, thanks to some Data Cleansing code
* make the data available to downstream exploration and analytics processes

### Data Exploration
* Step 1: Load data
* Step 2: Explore data
* Step 3: In-depth statistical measures and visualizations
This module provides statistics and visualization on Data Set to identify good columns for modelling, data quality issues (e.g. missing values, ...) and anticipate potential feature transformations necessary. Assess how relevant is a certain measurement (e.g. use correlation matrix) Get an idea on the value distribution of your data using statistical measures and visualizations.
It could seem that Data Exploration should be done before the data ETL, but here exploration is intended to understand completely the data, either before and after each cleaning process. So it is useful to fo it after the data has been loaded and formatted to be used in the framework.
This data exploration step requires, 
Explore and visualize features.
<!---
TODO
After the data model, it can be carried on: intelligent forecasting, advanced analytics, and exploratory visualization
---> 
### Feature Engineering
Contains the preprocessing part.
This part: change, enhance data for building the model
Data transformation as source data does not fit what model needs all the time.
Ideally, we have clean data but I never get it.
You may say that we should have a data engineering team helps on data transformation.
However, we may not know what we need understudying data.
One of the important requirement is both off-line training and online prediction should use the same pipeline to reduce misalignment.
### Modeling
Model building such as tackling classification problem.
It should not just include model training part but also an evaluation part.
On the other hand, we have to think about multiple models scenario.
A typical use case is an ensemble model such as combing Logistic Regression model and Neural Network model.

Finally export result (and data visualization)
<!---
TODO
This part should be useful for both reporting and analytics.
---> 

## Version information

Minium system reqirment:
* Python 3.4
* Tensorflow (>=1.14.0,<2.0.0)


As for Pathlib, better to use Python 3.6 (>3.4), the pathlib module is supported throughout the standard library, partly due to the addition of a file system path protocol [see](https://realpython.com/python-pathlib/).
<!---
The Better Solution: Python 3’s pathlib!
Python 3.4 introduced a new standard library for dealing with files and paths called pathlib — and it’s great!
---> 





# Folder review

This is the final Project Organization:
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for users looking at this project.
    ├── data
    │   ├── intermediate   <- Intermediate data that has been transformed.
    │   ├── log_files      <- Log processing files
    │   ├── models         <- Trained and serialized models, model predictions, or model summaries
    │   ├── processed      <- The final, canonical data sets for modelling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── reports            <- Any output generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── cup_project        <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data_etl       <- Scripts to extract transform and load  data
    │   │
    │   ├── data_exp       <- Scripts to explore, process and clean data
    │   │
    │   ├── features_eng   <- Scripts to turn cleaned data into features for modelling
    │   │   └── build_features.py
    │   │
    │   ├── model_def      <- Scripts to train models and then use trained models to make
    │   │
    │   └── model_trn_eval <- Scripts to create exploratory and results-oriented visualizations
    │
    └── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported

--------






# References
Project structure ideas:
* Edward Ma, [Manage your Data Science project structure in early stage](https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600), Sep 22, 2018
* [Cookiecutter Data Science project template](https://drivendata.github.io/cookiecutter-data-science/)
* [Software Engineering for ML](https://www.comet.ml/site/why-software-engineering-processes-and-tools-dont-work-for-machine-learning/)
* [How to Build A Data Set For Your Machine Learning Project](https://towardsdatascience.com/how-to-build-a-data-set-for-your-machine-learning-project-5b3b871881ac)
* [Data, structure, and the data science pipeline](https://developer.ibm.com/articles/ba-intro-data-science-1/)
* [The lightweight IBM Cloud Garage Method](https://developer.ibm.com/articles/the-lightweight-ibm-cloud-garage-method-for-data-science/)
* [Architectural decisions guidelines](https://developer.ibm.com/articles/data-science-architectural-decisions-guidelines/)
* [Architectural thinking](https://developer.ibm.com/technologies/artificial-intelligence/articles/architectural-thinking-in-the-wild-west-of-data-science/)
* [Data, Analytics and AI](https://www.ibm.com/cloud/architecture/architectures/dataAIArchitecture)
* [How to plan and execute your ML and DL projects](https://blog.floydhub.com/structuring-and-planning-your-machine-learning-project/)
* [Becoming One With the Data](https://blog.floydhub.com/becoming-one-with-the-data/)
* [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)
* [Step-by-Step Guide to Creating R and Python Libraries](https://towardsdatascience.com/step-by-step-guide-to-creating-r-and-python-libraries-e81bbea87911)
