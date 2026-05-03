import streamlit as st
import pandas as pd
import numpy as np
import time  # we'll use this later
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


df1 = pd.read_csv("ufcdat/UFC dataset/Fighter stats/fighter_stats.csv")

def choose_fighter(age):
    filtered_fighters = df1[df1['age'] == float(age)]
    
    if filtered_fighters.empty:
        return f"No fighters of age {age}"
    
    selected_fighter = filtered_fighters.sample(n=1).iloc[0]
    return f"Randomly selected fighter of age {age}: {selected_fighter['name']}"

def choose_fighter_name(name):
    filtered_fighters = df1[df1['name'] == name]
    
    if filtered_fighters.empty:
        return f"No fighters named {name}"
    
    selected_fighter = filtered_fighters.sample(n=1).iloc[0]
    return f"There is indeed a fighter named {name}: {selected_fighter['name']}"

def choose_fighter_stance(stance):
    filtered_fighters = df1[df1['stance'] == stance]
    
    if filtered_fighters.empty:
        return f"No fighters with {stance} stance"
    
    selected_fighter = filtered_fighters.sample(n=1).iloc[0]
    return f"Randomly selected fighter of {stance} stance: {selected_fighter['name']}"

def choose_fighter_stance_age(age,stance):
    filtered_fighters = df1[(df1['age'] == float(age)) & (df1['stance'] == stance)]
    
    if filtered_fighters.empty:
        return f"No fighters of age {age} and {stance} stance"
    
    selected_fighter = filtered_fighters.sample(n=1).iloc[0]
    return f"Randomly selected fighter of age {age} and stance {stance}: {selected_fighter['name']}"

print(df1.head())
st.title("My Awesome UFC Statistics Dashboard!")
st.subheader("An analysis of the fighter stats...")
st.write("This app displays many interactive elements that give insight into the data I analyzed.")

st.header("Some little fun tools")

age = st.slider("Enter an age:", 0,100)
# Dropdown menu
stance = st.selectbox("Choose a fighting stance:", ["Orthodox", "Southpaw"])

st.write(f"{choose_fighter(age)}")
st.write(f"{choose_fighter_stance(stance)}")
st.write(f"{choose_fighter_stance_age(age,stance)}")

name = st.text_input("Enter fighter name:", "Enter name")
st.write(f"{choose_fighter_name(name)}")

height = df1['height']
reach_height = df1[['reach','height']]
weight = df1['weight']

st.scatter_chart(df1, x="height", y="reach",x_label = 'Height(in.)',y_label='Reach(in.)')

# Drop rows with NaN in height or reach
df1_clean = df1.dropna(subset=['height', 'reach'])

ufc_train, ufc_test = train_test_split(df1_clean, test_size=0.2, random_state=2026)
x_train = ufc_train[['height']]
x_test = ufc_test[['height']]
y_train = ufc_train[['reach']]
y_test = ufc_test[['reach']]

linreg = LinearRegression().fit(x_train, y_train)
train_r2 = linreg.score(x_train, y_train)
test_r2 = linreg.score(x_test, y_test)
train_mse = mean_squared_error(linreg.predict(x_train), y_train)
test_mse = mean_squared_error(linreg.predict(x_test), y_test)
# Visualize Linear Model Results
x_lin = np.linspace(df1_clean['height'].min(), df1_clean['height'].max(), 100).reshape(-1, 1)
preds = linreg.predict(x_lin)

fig, ax = plt.subplots()
plt.xlabel("Height")
plt.ylabel("Reach")
plt.title("Zoomed in graph of height and reach")
ax.scatter(df1_clean["height"], df1_clean["reach"])
ax.plot(x_lin, preds, color='red')

st.pyplot(fig)

st.subheader(f"Line equation is: y = {linreg.coef_[0][0]:.4f}x {linreg.intercept_[0]:.4f}")
st.subheader(f'Training R2 Score: {train_r2:.4f}')
st.subheader(f'Test R2 Score: {test_r2:.4f}')


st.image("https://www.thescore.com/mma/news/1050608", caption="GSP Cool as Hell")

# Embed a video
st.video("https://www.youtube.com/watch?v=Li_X1dQtP34&t=297s")

