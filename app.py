import streamlit as st
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #ff9933;
        color: white;
        font-weight: bold;
    }
    .stSelectbox {
        background-color: #ffffff;
    }
    .stHeader {
        color: #000080;
    }
</style>
""", unsafe_allow_html=True)

teams = ['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Delhi Capitals', 'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants',
         'Sunrisers Hyderabad', 'Rajasthan Royals', 'Kolkata Knight Riders']

cities = ['Hyderabad', 'Rajkot', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
          'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town',
          'Port Elizabeth', 'Durban', 'Centurion', 'East London',
          'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
          'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi', 'Visakhapatnam',
          'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali',
          'Bengaluru']

pipe_path = os.path.join(os.path.dirname(__file__), 'pipe.pkl')
pipe = pickle.load(open(pipe_path, 'rb'))

st.title("🏏 IPL Win Predictor 🏆")

# Team selection
col1, col2 = st.columns(2)
with col1:
    st.write("Batting Team")
    batting_team = st.selectbox('Select the batting team', sorted(teams), key='batting_team')
with col2:
    st.write("Bowling Team")
    bowling_team = st.selectbox('Select the bowling team', sorted(teams), key='bowling_team')

# City selection
st.write("Host City")
selected_city = st.selectbox('Select Host City', sorted(cities), key='host_city')

# Match situation inputs
st.subheader("Match Situation")
col3, col4, col5, col6 = st.columns(4)
with col3:
    target = st.number_input('Target', min_value=0, max_value=300, value=150)
with col4:
    score = st.number_input('Current Score', min_value=0, max_value=300, value=0)
with col5:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, value=0.0, step=0.1)
with col6:
    wickets = st.number_input('Wickets Lost', min_value=0, max_value=10, value=0, step=1)

# Prediction
if st.button('Predict Win Probability', key='predict'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                             'city': [selected_city], 'runs_left': [runs_left],
                             'balls_left': [balls_left], 'wickets': [wickets],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Create a smaller pie chart
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.pie([win, loss], labels=[f'{batting_team}\n{win:.1%}', f'{bowling_team}\n{loss:.1%}'],
           autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    ax.axis('equal')
    st.pyplot(fig)

    # Display win probabilities with progress bars
    st.subheader("Win Probabilities")
    st.progress(win)
    st.write(f"{batting_team}: {win:.1%}")
    st.progress(loss)
    st.write(f"{bowling_team}: {loss:.1%}")

    # Additional stats
    st.subheader("Match Stats")
    col7, col8, col9 = st.columns(3)
    with col7:
        st.metric("Runs Needed", runs_left)
    with col8:
        st.metric("Run Rate Required", round(rrr, 2))
    with col9:
        st.metric("Overs Left", round((120 - (overs * 6)) / 6, 1))

# Footer
st.markdown("---")
st.header("Created by Shaurya Dobhal")
st.markdown("Data source: IPL")