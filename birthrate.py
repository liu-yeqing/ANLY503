#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 03:13:29 2019

@author: Lin
"""

import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from numpy import cov
import seaborn as sns
import scipy





################## Economic Analysis#####################

########## birth rate VS. inflation ##########

def line_bar(x,y,y1):
    # define colors
    colors=[]
    for i in y:
        if i >= 0:
            colors.append( 'green')
        else:
            colors.append( 'red')
            
    # plot bar and line        
    fig = plt.figure()
    ax = fig.gca() 
    ax.bar(x, y, align='center', alpha=0.5,color=colors)    
    ax.plot(x, y1)
    ax.title.set_text('Inflation and Birth Rate')
    plt.show()
    

###########################
          

def econAnalysis():
    inflation_raw=pd.read_csv('/Users/qq/Desktop/y2/visualization/proj/inflation_data.csv')
    gdp_raw=pd.read_csv('/Users/qq/Desktop/y2/visualization/proj/gdp_data.csv')
    b_rate_raw=pd.read_csv('/Users/qq/Desktop/y2/visualization/proj/econ_birthrate_data.csv')
    
    # set up a big table
    econ_data=pd.DataFrame()
    econ_data['year']=b_rate_raw['Year']
    econ_data['CHN_brate']=b_rate_raw['China']
    econ_data['CHN_gdp']=gdp_raw['China']
    #econ_data['CHN_debt']=debt_raw['China']
    econ_data['CHN_inflation']=inflation_raw['China']
    
    econ_data['US_brate']=b_rate_raw['United States']
    econ_data['US_gdp']=gdp_raw['United States']
    #econ_data['US_debt']=debt_raw['United States']
    econ_data['US_inflation']=inflation_raw['United States']
    
    econ_data['IND_brate']=b_rate_raw['India']
    econ_data['IND_gdp']=gdp_raw['India']
    #econ_data['IND_debt']=debt_raw['India']
    econ_data['IND_inflation']=inflation_raw['India']
    
    #econ_data['CHN_household_debt_amount']=econ_data['CHN_debt']*econ_data['CHN_gdp']*0.01
    #econ_data['IND_household_debt_amount']=econ_data['IND_debt']*econ_data['IND_gdp']*0.01
    #econ_data['USA_household_debt_amount']=econ_data['US_debt']*econ_data['US_gdp']*0.01
    
    econ_data=econ_data[econ_data['year']<=2017]
    
    # save to csv
    econ_data.to_csv('/Users/qq/Desktop/y2/visualization/proj/combined_whole.csv')
    

         
    a=econ_data[econ_data['year']>=1987]
    a=a[a['year']<=2017]
    year_CN=a['year']
    y_CN=a['CHN_inflation']  
    y1_CN=a['CHN_brate']
    line_bar(year_CN,y_CN,y1_CN) 
    
    year=econ_data['year']
    y_US=econ_data['US_inflation']  
    y1_US=econ_data['US_brate']
    line_bar(year,y_US,y1_US)
    
    year=econ_data['year']
    y_IN=econ_data['IND_inflation']  
    y1_IN=econ_data['IND_brate']
    line_bar(year,y_IN,y1_IN)
    
    
    
    
    ################# correlation ######################
    
    corr_CN,_=pearsonr(y_CN, y1_CN)
    corr_US,_=pearsonr(y_US, y1_US)
    corr_IN,_=pearsonr(y_IN, y1_IN)
    print('Pearsons correlation of inflation and birth rate: %.3f' % corr_CN)
    print('Pearsons correlation of inflation and birth rate: %.3f' % corr_US)
    print('Pearsons correlation of inflation and birth rate: %.3f' % corr_IN)
    
    
    
    gdp_CN=a['CHN_gdp']  
    br_CN=a['CHN_brate']
    
    
    gdp_US=econ_data['US_gdp']  
    br_US=econ_data['US_brate']
    
    gdp_IN=econ_data['IND_gdp']  
    br_IN=econ_data['IND_brate']
    
    corr_CN_gdp,_=pearsonr(gdp_CN, br_CN)
    corr_US_gdp,_=pearsonr(gdp_US, br_US)
    corr_IN_gdp,_=pearsonr(gdp_IN, br_IN)
    print('Pearsons correlation of gdp and birth rate: %.3f' % corr_CN_gdp)
    print('Pearsons correlation of gdp and birth rate: %.3f' % corr_US_gdp)
    print('Pearsons correlation of gdp and birth rate: %.3f' % corr_IN_gdp)
    ############################ economy end #############################
    
    
    
    
    
    
    
    
    
    
    
    
    
    ################# Feminism Analysis #3###########################

def fem_plot(x,y,y1,titleText):
    # define colors
    colors=[]
    for i in y:
        if i >= 0:
            colors.append( 'green')
        else:
            colors.append( 'red')
            
    # plot bar and line        
    fig = plt.figure()
    ax = fig.gca() 
    ax.bar(x, y, align='center', alpha=0.5,color=colors)   
    ax.plot(x, y1)
    
    ax.title.set_text(titleText)
    plt.show()
    

    
def feminismAnalysis():
        # load datasets
    employName = "LabourForce.xls"
    eduName = "SecondaryEnrol.xls"
    wageName = "WageGenderGap.xls"
    birthrateName = "birthrate.xls"
    employ = pd.read_excel(employName)
    edu = pd.read_excel(eduName)
    wage = pd.read_excel(wageName)
    birthrate = pd.read_excel(birthrateName)
    # datasets eda
    employ.head()
    edu.head()
    wage.head()
    
    eduUS = edu.loc[edu['Country Name'] == "United States"]
    eduCN = edu.loc[edu['Country Name'] == "China"]
    eduIND = edu.loc[edu['Country Name'] == "India"]
    
    eduframes = [eduUS,eduCN, eduIND]
    eduNEW = pd.concat(eduframes)
    eduNEW.set_index("Country Name", inplace = True)
    eduNEW = eduNEW.drop(['Country Code','Indicator Name','Indicator Code'],axis=1)
        
    #Similar data preprocessing applied on the employment data
    employUS = employ.loc[employ['Country Name'] == "United States"]
    employCN = employ.loc[employ['Country Name'] == "China"]
    employIND = employ.loc[employ['Country Name'] == "India"]
    employframes = [employUS,employCN,employIND]
    employNEW = pd.concat(employframes)
    employNEW.set_index("Country Name", inplace = True)
    employNEW =employNEW.drop(['Country Code','Indicator Name','Indicator Code'],axis=1)    
        
    #Similar data preprocessing applied on the employment data
    wageUS = wage.loc[employ['Country Name'] == "United States"]
    wageCN = wage.loc[employ['Country Name'] == "China"]
    wageIND = wage.loc[employ['Country Name'] == "India"]
    wageframes = [wageUS,wageCN,wageIND]
    wageNEW = pd.concat(wageframes)
    wageNEW.set_index("Country Name", inplace = True)
    wageNEW = wageNEW.drop(['Country Code','Indicator Name','Indicator Code'],axis=1)    
        
    birthUS = birthrate.loc[birthrate['Country Name'] == "United States"]
    birthCN = birthrate.loc[birthrate['Country Name'] == "China"]
    birthIND = birthrate.loc[birthrate['Country Name'] == "India"]
    birthframes = [birthUS,birthCN,birthIND]
    birthNEW = pd.concat(birthframes)
    birthNEW.set_index("Country Name", inplace = True)
    birthNEW = birthNEW.drop(['Country Code','Indicator Name','Indicator Code'],axis=1)
    birthNEW.head()    
        
    eduNEW = eduNEW.fillna(0).T
    employNEW = employNEW.fillna(0).T
    wageNEW = wageNEW.fillna(0).T
    birthNEW = birthNEW.fillna(0).T
    
    femaleData = pd.DataFrame()
    femaleData['eduUS'] = eduNEW['United States']
    femaleData['eduCN'] = eduNEW['China']
    femaleData['eduIND'] = eduNEW['India']
    femaleData['wageUS'] = wageNEW['United States']
    femaleData['wageCN'] = wageNEW['China']
    femaleData['wageIND'] = wageNEW['India']
    femaleData['birthUS'] = birthNEW['United States']
    femaleData['birthCN'] = birthNEW['China']
    femaleData['birthIND'] = birthNEW['India']
    femaleData.reset_index(level=0, inplace=True)
    femaleData.rename(columns={'index':'year'}, inplace=True)
    
    femaleData.to_excel("femaleData.xls")
    
    
    year=femaleData['year']
    y_US=femaleData['eduUS']  
    y1_US=femaleData['birthUS']
    #y2_US=femaleData['wageUS']
    titleText = ("US Female Education Rate VS Birth Rate")
    fem_plot(year,y_US,y1_US,titleText)
    
    y_CN=femaleData['eduCN']  
    y1_CN=femaleData['birthCN']
    titleText = ("China Female Education Rate VS Birth Rate")
    fem_plot(year,y_CN,y1_CN,titleText)
    
    y_IND=femaleData['eduIND']  
    y1_IND=femaleData['birthIND']
    titleText = ("India Female Education Rate VS Birth Rate")
    fem_plot(year,y_IND,y1_IND,titleText)
    
    
    
    ##### correlation
    
    ### correlation 
    # for Education rate 
    
    corr_CN,_=pearsonr(y_CN, y1_CN)
    corr_US,_=pearsonr(y_US, y1_US)
    corr_IND,_=pearsonr(y_IND, y1_IND)
    print('China Education Pearsons correlation: %.3f' % corr_CN)
    print('US Education Pearsons correlation: %.3f' % corr_US)
    print('India Education Pearsons correlation: %.3f' % corr_IND)
    
    # for wage level
    
    wage_CN = femaleData['wageCN']  
    wage_US = femaleData['wageUS']  
    wage_IND= femaleData['wageIND']  
    
    corr_CN_w,_=pearsonr(wage_CN, y1_CN)
    corr_US_w,_=pearsonr(wage_US, y1_US)
    corr_IND_w,_=pearsonr(wage_IND, y1_IND)
    
    print('China Wage Pearsons correlation: %.3f' % corr_CN_w)
    print('US Wage Pearsons correlation: %.3f' % corr_US_w)
    print('India Wage Pearsons correlation: %.3f' % corr_IND_w)
    
    




def healthcare():
    
        """## 1. Contraceptive Prevalence Dataset
    
    ### 1.1 Load data
    """
    
    CP = pd.read_csv('CP.csv',sep = ',', header = 0)
    
    CP.head()
    
    print('The number of rows: '+ str(CP.shape[0]))
    
    print('The number of columns: '+ str(CP.shape[1]))
    
    """### 1.2 Data exploration"""
    
    CP.set_index('Country Name',inplace = True) # set the index of dataframe to 'Counrty Name'
    
    CP.head()
    
    tctry_CP = CP.drop(['Country Code','Indicator Name','Indicator Code','2018','2019'], axis = 1).loc[['China','India',
    'United States']]
    
    tctry_CP
    
    print(tctry_CP.isnull().sum(axis = 1))
    
    """For each country, among 58 years, over 39 years has missing value."""
    
    tctry_CP.apply(pd.DataFrame.describe, axis=1)
    
    """### 1.3 Dealing with missing values"""
    
    tctry_CP.loc['China'] = tctry_CP.loc['China'].interpolate().fillna(method = 'bfill')
    
    tctry_CP.loc['India'] = tctry_CP.loc['India'].interpolate().fillna(method = 'bfill')
    
    tctry_CP.loc['United States'] = tctry_CP.loc['United States'].interpolate().fillna(method = 'bfill')
    
    tctry_CP
    
    CN_CP = tctry_CP.loc['China']
    
    IND_CP = tctry_CP.loc['India']
    
    US_CP = tctry_CP.loc['United States']
    
    CN_CP.index
    
    plt.figure(figsize=(16,10)) 
    plt.plot(CN_CP.index, CN_CP.values, 'r--', IND_CP.index, IND_CP.values, 'bs',
            US_CP.index, US_CP.values, 'g^')
    plt.legend(('China', 'India', 'United States'),loc='upper right', shadow=True)
    plt.xticks(rotation=50)
    plt.xlabel('Year')
    plt.ylabel('Rate')
    plt.title('Contraceptive Prevalence in China, India and USA from 1960 to 2017')
    
    """## 2. Mortality Rate Dataset
    
    ### 2.1 Load data
    """
    
    MR = pd.read_csv('MR.csv', sep = ',', header = 0)
    
    MR.head()
    
    print('The number of rows: '+ str(MR.shape[0]))
    
    print('The number of columns: '+ str(MR.shape[1]))
    
    """### 2.2 Data exploration"""
    
    MR.set_index('Country Name',inplace = True) # set the index of dataframe to 'Counrty Name'
    
    MR.head()
    
    tctry_MR = MR.drop(['Country Code','Indicator Name','Indicator Code','2018','2019'], axis = 1).loc[['China','India',
    'United States']]
    
    tctry_MR
    
    print(tctry_MR.isnull().sum(axis = 1))
    
    """Among three countries, only China has 9 missing values"""
    
    tctry_MR.apply(pd.DataFrame.describe, axis=1)
    
    CN_MR = tctry_MR.xs('China')
    CN_MR_plot = CN_MR.plot(kind = 'line',colormap='Reds_r',figsize = (16,10),
    title = 'Mortality Rate of Infant (per 1000 live births) in China from 1960 to 2017')
    CN_MR_plot.set_xlabel("Year")
    CN_MR_plot.set_ylabel("Rate")
    CN_MR_plot2 = CN_MR.plot(kind = 'bar',colormap='Reds')
    
    """There are missing values for China from 1960 to 1968"""
    
    IND_MR = tctry_MR.xs('India')
    IND_MR_plot = IND_MR.plot(kind = 'line',colormap='Greens_r',figsize = (16,10),
    title = 'Mortality Rate of Infant (per 1000 live births) in India from 1960 to 2017')
    IND_MR_plot.set_xlabel("Year")
    IND_MR_plot.set_ylabel("Rate")
    IND_MR_plot2 = IND_MR.plot(kind = 'bar',colormap='Greens')
    
    US_MR = tctry_MR.xs('United States')
    US_MR_plot = US_MR.plot(kind = 'line',colormap='Blues_r',figsize = (16,10),
    title = 'Mortality Rate of Infant (per 1000 live births) in US from 1960 to 2017')
    US_MR_plot.set_xlabel("Year")
    US_MR_plot.set_ylabel("Rate")
    US_MR_plot2 = US_MR.plot(kind = 'bar',colormap='Blues')
    
    """We can know from three graphs above that there is a trend for each country: as year goes by, the mortality rate of infant decreases.
    
    ### 2.3 Dealing with missing values
    
    Since this dataset is a time series data, we can handle missing data of China by using linear interpolation.
    """
    
    tctry_MR.loc['China'].fillna(method = 'bfill',inplace = True) # Next observation carried backward
    
    tctry_MR
    
    NewCN_MR = tctry_MR.loc['China']
    
    print(NewCN_MR)
    
    fig = plt.figure(figsize = (16,10))
    ax  = fig.add_subplot(111) 
    ax.set_xlabel('Country',fontsize = 13)
    ax.set_ylabel('Rate',fontsize = 13)
    new_boxplot = ax.boxplot(tctry_MR,patch_artist = True, labels = ('China','India','United States'))
    colors = ['tan', 'pink', 'green']
    
    for patch, color in zip(new_boxplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.figure(figsize=(16,10)) 
    plt.plot(NewCN_MR.index,NewCN_MR.values, 'r--', NewCN_MR.index, IND_MR.values, 'bs',
            NewCN_MR.index, US_MR.values, 'g^')
    plt.legend(('China', 'India', 'United States'),loc='upper right', shadow=True)
    plt.xticks(rotation=50)
    plt.xlabel('Year')
    plt.ylabel('Rate')
    plt.title('Mortality Rate of Infant (per 1000 live births) in China, India and USA from 1960 to 2017')
    
    """## 3. Relationship between birthrate and healthcare (contraceptive prevalence and mortality rate)
    
    ### 3.1 Load birthrate dataset
    """
    
    BR = pd.read_csv('birthrate.csv', sep = ',',header = 0)
    
    BR.head()
    
    BR.set_index('Country Name',inplace = True)
    
    tctry_BR = BR.drop(['Country Code','Indicator Name','Indicator Code','2018','2019'], axis = 1).loc[['China','India','United States']]
    
    tctry_BR
    
    print(tctry_BR.isnull().sum(axis = 1))
    
    """For each country, there is no missing values"""
    
    tctry_BR.apply(pd.DataFrame.describe,axis = 1)
    
    """### 3.2 Merge three datasets"""
    
    tctry_CP
    
    tctry_MR
    
    tctry_BR
    
    BrHealth_data = pd.DataFrame()
    
    BrHealth_data['Year'] = tctry_BR.iloc[0].index
    
    BrHealth_data['CN_BR'] = tctry_BR.loc['China'].values.round(2)
    
    BrHealth_data['CN_CP'] = tctry_CP.loc['China'].values.round(2)
    
    BrHealth_data['CN_MR'] = tctry_MR.loc['China'].values.round(2)
    
    BrHealth_data['IND_BR'] = tctry_BR.loc['India'].values.round(2)
    
    BrHealth_data['IND_CP'] = tctry_CP.loc['India'].values.round(2)
    
    BrHealth_data['IND_MR'] = tctry_MR.loc['India'].values.round(2)
    
    BrHealth_data['US_BR'] = tctry_BR.loc['United States'].values.round(2)
    
    BrHealth_data['US_CP'] = tctry_CP.loc['United States'].values.round(2)
    
    BrHealth_data['US_MR'] = tctry_MR.loc['United States'].values.round(2)
    
    BrHealth_data.head()
    
    #BrHealth_data.to_csv('Br_health.csv')
    
    """### 3.3 Correlation between birth rate and healthcare (two features) for each country
    
    #### 3.3.1 China
    """
    
    # correlation between birth rate and contraceptive prevalence
    CN_BrcorrCp, _ = pearsonr(BrHealth_data['CN_BR'], BrHealth_data['CN_CP'])
    print('Pearsons correlation: %.3f' % CN_BrcorrCp)
    
    """The correlation is -0.744, which means that the birth rate of China has negative correlation with contraceptive prevalence"""
    
    # correlation between birth rate and mortality rate
    CN_BrcorrMr, _ = pearsonr(BrHealth_data['CN_BR'], BrHealth_data['CN_MR'])
    print('Pearsons correlation: %.3f' % CN_BrcorrMr)
    
    """The correlation is 0.894, which means that the birth rate of China has positive correlation with mortality rate of infant
    
    #### 3.3.2 India
    """
    
    # correlation between birth rate and contraceptive prevalence
    IND_BrcorrCp, _ = pearsonr(BrHealth_data['IND_BR'], BrHealth_data['IND_CP'])
    print('Pearsons correlation: %.3f' % IND_BrcorrCp)
    
    """The correlation is -0.927, which means that the birth rate of India has highly negative correlation with contraceptive prevalence"""
    
    # correlation between birth rate and mortality rate
    IND_BrcorrMr, _ = pearsonr(BrHealth_data['IND_BR'], BrHealth_data['IND_MR'])
    print('Pearsons correlation: %.3f' % IND_BrcorrMr)
    
    """The correlation is 0.988, which means that the birth rate of India has positive correlation with mortality rate of infant
    
    #### 3.3.3 United States
    """
    
    # correlation between birth rate and contraceptive prevalence
    US_BrcorrCp, _ = pearsonr(BrHealth_data['US_BR'], BrHealth_data['US_CP'])
    print('Pearsons correlation: %.3f' % US_BrcorrCp)
    
    """The correlation is -0.751, which means that the birth rate of United States has negative correlation with contraceptive prevalence"""
    
    # correlation between birth rate and mortality rate
    US_BrcorrMr, _ = pearsonr(BrHealth_data['US_BR'], BrHealth_data['US_MR'])
    print('Pearsons correlation: %.3f' % US_BrcorrMr)
    
    """The correlation is 0.878, which means that the birth rate of United States has positive correlation with mortality rate of infant
    
    For each country, the birth rate is negative correlated to contraceptive prevalence. As the contraceptive methods are more prevalent, the birth rate is decreasing. Also, the birth rate is highly positive correlated to mortaility rate of infant. As mortality rate goes down, the birth rate decreases as well. Since low mortality rate of infant means healthcare improves, we can conclude that as healthcare developes, people tend to have less children.
    
    ### 3.4 linear regression
    """


    
    
    
    

