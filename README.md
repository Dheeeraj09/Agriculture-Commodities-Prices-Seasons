# Challenge 1: Agriculture Commodities, Prices & Seasons

## Aim

Your team is working on building a variety of insight packs to measure key trends in the
Agriculture sector in India. You are presented with a data set around Agriculture and your aim is
to understand trends in APMC (Agricultural produce market committee)/mandi price & quantity
arrival data for different commodities in Maharashtra.

## Objectives

1. Test and filter outliers.
2. Detect seasonality type (multiplicative or additive) for each cluster of APMC and
commodities
3. De-seasonalise prices for each commodity and APMC according to the detected
seasonality type
4. Compare prices in APMC/Mandi with MSP(Minimum Support Price)- raw and
deseasonalised
5. Flag set of APMC/mandis and commodities with highest price fluctuation across different
commodities in each relevant season, and year.

## Results


**Final result files descriptions**
1) `Agriculture_Commodities_Prices & Seasons_analysis.ipynb`: This notebook contain all the analysis, graphs and charts along with proper explanation of methods and code of each step to perform the given tasks.; 
2) `outlier_removed_data.csv`: This csv file contain the data after removing the outliers; 
3) `preprocess_data.csv`: This csv file contain the preprocessed data used while detecting type of seasonality and then depersonalized it; 
4) `deseasonalised_data.csv`: This csv file contain the depersonalized data;
5) `Seasonal_type_list.csv`: This csv file contain the information of type of seasonality present in different commodities;
6) `flag_set_data_year.csv`: This csv file contain the highest price fluctuation information across different commodities in each relevant year.; 
7) `flag_data_year_season.csv`: This csv file contain the highest price fluctuation information across different commodities in each relevant year and season.
 
