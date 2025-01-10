import streamlit as st
import os
import pickle
import pandas as pd


from ipywidgets import Password

teams = ['Chennai Super Kings',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Delhi Capitals',
 'Punjab Kings',
 'Gujrat Titans',
 'Lucknow Supergaints',
 'Sunrisers Hyderabad',
 'Rajasthan Royals',
 'Kolkata Knight Riders']

cities = ['Hyderabad', 'Rajkot', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
       'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town',
       'Port Elizabeth', 'Durban', 'Centurion', 'East London',
       'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
       'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi', 'Visakhapatnam',
       'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali',
       'Bengaluru']

pipe_path = os.path.join(os.path.dirname(__file__), 'pipe.pkl')
pipe = pickle.load(open(pipe_path, 'rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select Host City',sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs Completed')
with col5:
    wickets = st.number_input('Wickets Out')

if st.button('Predict Probablity'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],
                             'bowling_team':[bowling_team],
                             'city':[selected_city],
                             'runs_left':[runs_left],
                             'balls_left':[balls_left],
                             'wickets':[wickets],
                             'total_runs_x':[target],
                             'crr':[crr],'rrr':[rrr]})
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + " - " + str(round(win*100)) + "%")
    st.header(bowling_team + " - " + str(round(loss*100)) + "%")