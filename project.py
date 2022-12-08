import streamlit as st

st.markdown("# Are you a Linkedin User?")

name = "Adam Brand Programming II Final"

st.write(name)

st.markdown("---")

import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


ss = pd.DataFrame({
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": clean_sm(s["par"]),
    "married": clean_sm(s["marital"]),
    "female": clean_sm(s["gender"]),
    "age": np.where(s["age"] > 98, np.nan, s["age"]),
    "sm_li": clean_sm(s["web1h"])})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    test_size = 0.2,
                                                    random_state = 555)


lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train, y_train)

answer_income = st.selectbox(label = "What is your level of income?",
options = ("Less than $10,000", "$10k - $20k", "$20k - $30k", "$30k - $40k", "$40k - $50k",
            "$50k - $75k", "$75k - $100k", "$100k - $150k", "$150k or more"))
if answer_income == "Less than $10,000": answer_income = 1
elif answer_income == "$10k - $20k": answer_income = 2
elif answer_income == "$20k - $30k": answer_income = 3
elif answer_income == "$30k - $40k": answer_income = 4
elif answer_income == "$40k - $50k": answer_income = 5
elif answer_income == "$50k - $75k": answer_income = 6
elif answer_income == "$75k - $100k": answer_income = 7
elif answer_income == "$100k - $150k": answer_income = 8
else: answer_income = 9


answer_education = st.selectbox(label = "What is your level of education?",
options = ("Less than Highschool", "Some Highschool", "Highschool Graduate", 
            "Some College", "Associate's Degree", "Bachelor's Degree", "Some Postgraduate",
            "Postgraduate Degree"))
if answer_education == "Less than Highschool": answer_education = 1
elif answer_education == "Some Highschool": answer_education = 2
elif answer_education == "Highschool Graduate": answer_education = 3
elif answer_education == "Some College": answer_education = 4
elif answer_education == "Associate's Degree": answer_education = 5
elif answer_education == "Bachelor's Degree": answer_education = 6
elif answer_education == "Some Postgraduate": answer_education = 7
else: answer_education = 8


answer_married = st.selectbox(label = "Are you married?",
options = ("No", "Yes"))
if answer_married == "No": answer_married = 0
else: answer_married = 1


answer_parent = st.selectbox(label = "Are you a Parent?",
options = ("No", "Yes"))
if answer_parent == "No": answer_parent = 0
else: answer_parent = 1


answer_female = st.selectbox(label = "Do you identify as one of Male or Female?",
options = ("Male", "Female"))
if answer_female == "Male": answer_female = 0
else: answer_female = 1


answer_age = st.slider(label = "Age", value = 40)


predict_if_user = pd.DataFrame({
    "income": [answer_income],
    "education": [answer_education],
    "parent": [answer_parent],
    "married": [answer_married],
    "female": [answer_female],
    "age": [answer_age]
})

prediction = lr.predict(predict_if_user)
if prediction <= .50: prediction = "not a Linkedin user"
else: prediction = "a Linkedin user"

probabilityUser = [answer_income, answer_education, answer_parent, answer_married, 
    answer_female, answer_age]

probs = lr.predict_proba([probabilityUser])

st.markdown("#### I predict that you are...")

if st.button("Predict"):
    st.write(prediction)
    print(st.text(f"Probability of being a Linkedin user: {probs[0][1]}"))
else: st.write("")
