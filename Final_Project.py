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
import plotly.express as px

st.title("Math 10 Final Project")



df = pd.read_csv("/Users/jimmy/Downloads/draftkings_contest_history_(11-23-21).csv", na_values=" ", )
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




st.write("### I am basing my project on a dataset that has been tracking my bets on the Draftkings platform. I am an avid sports fan and like to have my skin in the games I watch and I have never been able to see what my performance has been like.")
df= df[["Sport","Game_Type","Entry","Date","Place","Contest_Entries","Place%","Winnings_Non_Ticket","Entry_Fee","Net"]].copy()




st.write("##### I would like to first trim down the dataset to the columns that I believe are most understandable and useful for this project. I will also convert numerical entry columns to such if possible.",
         df)


st.write("Some notable facts that I extracted from this data shows that I am relatively successful in this venture",
         "For example, to get my total profit, I summed up the 'Net' column and got $",
         df["Net"].sum(),  
         
         "This is all with me only entering contests of at most $",df["Entry_Fee"].max(),
         "while having my biggest winnings being of $",df["Winnings_Non_Ticket"].max(),
         "It is cool to see how much success I have with relatively low stakes and a win rate of",
         (((df["Net"]>0).sum())/ (len(df)))*100, '%.')
pie_chart = px.pie(df, values=df['Entry_Fee'].value_counts(), names= df['Entry_Fee'].unique())
st.write('To justify my claim of playing at relatively low stakes, here is the pie chart presenting all the entry fees and percentages at which I have played them at.',
         pie_chart)




st.write("##### To observe my levels of success of each sport, we present a bar graph of all the net values per day for each specific sport along with the sport's scatterplot")
sports =df['Sport'].unique()
selected_sport=st.selectbox('Select sport', sports, 4)
           
sport_chart1 =alt.Chart(df[df['Sport']==selected_sport]).mark_bar().encode(
    x = "Date",
    y = "Net",
    color = "Sport",
    tooltip= ['Date','Net']
).interactive()
sport_chart = px.scatter(df[df['Sport']==selected_sport], x="Entry_Fee", y="Net", color="Sport", marginal_y="violin",
           marginal_x="box")
sport_chart1
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
    color = alt.Color('Sport',scale=alt.Scale(scheme="rainbow")),
    tooltip=['Sport', 'Entry_Fee', 'Winnings_Non_Ticket', 'Place%' ]
).properties(
    width = 600,
    height = 500,
)


st.write(reg_org + reg_org.transform_regression('Place%', 'Winnings_Non_Ticket').mark_line())

reg = LinearRegression()

X= df['Place%'].values.reshape(-1,1)
y= df['Winnings_Non_Ticket'].values.reshape(-1,1)

linear_model.LinearRegression().fit(X,y)

reg.fit(X,y)
st.write('The common themes with most of my scatterplots are that there exist two outliers and since my winning percentage is 24%, most of data is crowded towards the x-axis.'
         'With that said the regression line from my data projects a coeficient of',
         reg.coef_,
         "for the 'place%' variable with an intercept of",
         reg.intercept_,
         'to project winnings based on my placement relative to the field.')
st.latex(r'''
         \widehat{Winnings} = -4.0835(Place Percentage)_i + 2.8687  
         ''')

st.write('If I were to remove the two outlier points we would observe that the projected winnings would decresse but not by much since most of the data is already clustered towards the bottom.',
         'The regression line for this case would also have a flatter decline with the absense of the outliers')
st.latex(r'''
         \widehat{Winnings} = -1.6041(Place Percentage)_i + 1.1782
         ''')


df_no_outliers= df[df['Winnings_Non_Ticket'] != 1000]
reg_alt= alt.Chart(df_no_outliers).mark_circle(size=60).encode(
    x='Place%',
    y='Winnings_Non_Ticket',
    color = alt.Color('Sport',scale=alt.Scale(scheme="rainbow")),
    tooltip=['Sport', 'Entry_Fee', 'Winnings_Non_Ticket', 'Place%' ]
)
st.write(reg_alt + reg_alt.transform_regression('Place%', 'Winnings_Non_Ticket').mark_line())

reg = LinearRegression()

X= df_no_outliers['Place%'].values.reshape(-1,1)
y= df_no_outliers['Winnings_Non_Ticket'].values.reshape(-1,1)

linear_model.LinearRegression().fit(X,y)

reg.fit(X,y)
# reg.coef_
# reg.intercept_


st.write('Next is a the result of an attempt to use a nueral network to fit ny data. The network consists of 3 layers and it does not look to be overfitting my dataset.'
         'A reason I believe my data is difficult to fit is because there is no real pattern to what makes my bets win or not since they are practically an attempt on predicting the future.')
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

scaler = StandardScaler()
scaler.fit(df[numeric_cols])

df[numeric_cols] = scaler.transform(df[numeric_cols])
X_train = df[numeric_cols]
y_train = df["Winnings_Non_Ticket"]
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (6,)),
        #keras.layers.Flatten(),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(1,activation="sigmoid")
    ]
    )

model.compile(
    loss="binary_crossentropy", 
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)

history = model.fit(X_train,y_train,epochs=100, validation_split = 0.2)

fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
fig







st.write('I have implemented new types of charts throughout the project with the plotly library. Thank you for your time. The link for my Github repository is below')
st.write('https://github.com/jimmyk2/Final_Project',unsafe_allow_html=True)


