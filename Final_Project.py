#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:12:14 2021

@author: jimmy
"""

import streamlit as st
import numpy as np
import altair as alt
import sklearn
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress most warnings in TensorFlow
from pandas.api.types import is_numeric_dtype
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

st.title("Math 10 Final Project")


df = pd.read_csv("https://raw.githubusercontent.com/jimmyk2/Final_Project/main/draftkings_contest_history_(11-23-21).csv", na_values=" ", )
df["Date"]= df["Contest_Date_EST"].map(lambda x:x[:-5])
df["Date"]= pd.to_datetime(df["Date"])
df['Year']=df['Date'].dt.year
df=df[df['Year']==2021]
df["Date"]= [i.date() for i in df['Date']]

x_1 = df["Winnings_Non_Ticket"].map(lambda s: s.replace(",",""))
x_1= pd.to_numeric(x_1.map(lambda s: s.replace("$","")))
df["Winnings_Non_Ticket"]= x_1

df["Entry_Fee"] = pd.to_numeric(df["Entry_Fee"].map(lambda s: s.replace("$","")))

df['Place%'] = df['Place'] / df['Contest_Entries']
df["Net"]= df["Winnings_Non_Ticket"]- df["Entry_Fee"]




st.write("### I am basing my project on a dataset that has been tracking my bets on the Draftkings platform. I am an avid sports fan and like to have my skin in the games I watch and I have never been able to see what my performance has been like.",df)
df= df[["Sport","Game_Type","Entry","Date","Place","Contest_Entries","Place%","Winnings_Non_Ticket","Entry_Fee","Net"]].copy()




st.write("##### I would like to first trim down the dataset to the columns that I believe are most understandable and useful for this project. I will also convert numerical entry columns to such if possible.",
         df)


st.write("Some notable facts that I extracted from this data shows that I am relatively successful in this venture",
         "For example, to get my total profit, I summed up the 'Net' column and got $",
         df["Net"].sum(),  
         
         "This is all with me only entering contests of at most $",df["Entry_Fee"].max(),
         "while having my biggest winnings being of $",df["Winnings_Non_Ticket"].max(),
         "It is cool to see how much success I have with relatively low stakes and a win rate of",
         (((df["Net"]>0).sum())/ (len(df)))*100, '%.',
         df["Entry_Fee"].value_counts())


st.write('##### To observe my levels of success of each sport, we present the bar graph of the net values every day for each specific sport.')
sports =df['Sport'].unique()
selected_sport=st.selectbox('Select sport', sports, 4)
sport_chart =alt.Chart(df[df['Sport']==selected_sport]).mark_bar().encode(
    x = "Date",
    y = "Net",
    color = "Place",
    tooltip= ['Date','Net']
).interactive()
sport_chart


st.write('##### We can also look at all bets on a certain day')
pick_date= st.date_input('',
              datetime.date(2021, 9, 21),
              datetime.date(2021, 4, 1),
              datetime.date(2021, 11, 23)
              )
st.write("Date",pick_date)
df[df['Date']==pick_date]

st.write('### Linear Regression')


reg_org= alt.Chart(df).mark_circle(size=60).encode(
    x='Place%',
    y='Winnings_Non_Ticket',
    color='Sport',
    tooltip=['Sport', 'Entry_Fee', 'Winnings_Non_Ticket', 'Place', 'Contest_Entries', ]
).properties(
    width = 600,
    height = 500,
).interactive()
reg_org


reg = LinearRegression()

X= df['Place%'].values.reshape(-1,1)
y= df['Winnings_Non_Ticket'].values.reshape(-1,1)

linear_model.LinearRegression().fit(X,y)

reg.fit(X,y)
reg.coef_
reg.intercept_

reg_no_last= alt.Chart(df[df['Place%']<1]).mark_circle(size=60).encode(
    x='Place%',
    y='Winnings_Non_Ticket',
    color='Sport',
    tooltip=['Sport', 'Entry_Fee', 'Winnings_Non_Ticket', 'Place', 'Contest_Entries', ]
).properties(
    width = 600,
    height = 500,
).interactive()
reg_no_last

reg = LinearRegression()
df_reg= df[df['Place%']<1]

X= df_reg['Place%'].values.reshape(-1,1)
y= df_reg['Winnings_Non_Ticket'].values.reshape(-1,1)

linear_model.LinearRegression().fit(X,y)

reg.fit(X,y)
reg.coef_
reg.intercept_





reg_alt= alt.Chart(df[df['Winnings_Non_Ticket']!=1000]).mark_circle(size=60).encode(
    x='Place%',
    y='Winnings_Non_Ticket',
    color='Sport',
    tooltip=['Sport', 'Entry_Fee', 'Winnings_Non_Ticket', 'Place', 'Contest_Entries', ]
)
reg_alt












