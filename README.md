# Virtua Fighter Machine Learning
An introduction to Machine Learning with data from Virtua Fighter matches

## Getting Started
**1 - Get Data**

The fastest way to get started is to download existing match data scraped by VirtuAnalytics. 

[Click here](https://drive.google.com/file/d/1aXf81emse3jqE2f93-b_v-B3tGd2jn3d/view?usp=sharing) to download the CSV data from web browser, or copy the below command into your terminal to download from there.

```
# Download data using command line
$ curl https://drive.google.com/file/d/1aXf81emse3jqE2f93-b_v-B3tGd2jn3d/view?usp=sharing

```

**2 - Train Using The Data**
```
$ python demos/train_logic.py --filename="vf_match_data.csv.gz"
```

**3 - Test The Results**
