import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid')
import joblib 
import streamlit as st
from geopy.geocoders import Nominatim
import math
from sklearn.preprocessing import LabelEncoder



model = joblib.load('model.h5')

st.title("Restaurants Success Predictor")


def DistanceFromCapital (city) :
        # Import the required library
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="MyApp")
    nom=Nominatim(domain='localhost:8080', scheme='http')
    location1 = geolocator.geocode("Bangalore")
    location2 = geolocator.geocode(city)

#     print("The latitude of the location1 is: ", location1.latitude)
#     print("The longitude of the location1 is: ", location1.longitude)
#     print("The latitude of the location2 is: ", location2.latitude)
#     print("The longitude of the location2 is: ", location2.longitude)
    lat1 = location1.latitude
    lon1 = location1.longitude 
    lat2 = location2.latitude
    lon2 = location2.longitude
    earth = 6371 #Earth's Radius in Kms.
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = int(earth * c)
    return d


def devidePhoneNumber (number) :
    number = str(number)
    if number.startswith('080') : 
        return 3
    elif number.startswith('+91') or number.startswith('00 91') : 
        return 2 
    else : 
        return 1


def dishlikesnumber(dishes) :
    dishes = dishes.lower()
    if  'no' in dishes  : 
        return 0 
    elif ',' not in dishes or ' ' in dishes : 
        return 1 
    elif ',' in dishes  : 
        return len(dishes.split(','))


def OnlineOrder(order) : 
    if order == 'Yes' : 
        return [0,1]
    elif order == 'No' : 
        return [1,0]

def BookTable(book) : 
    if book == 'Yes' : 
        return [0,1]
    else : 
        return [1,0] 

    
    

votes = st.number_input('Number Of visits to this Restaurant',0,100000)
approx_cost_For_two_people = st.number_input('approx_cost(for two people)' ,0,1000000,step =50)
Restaurant_City= st.text_input('Restaurant City' , 'Bangalore')
Distace_from_capital = DistanceFromCapital(Restaurant_City)
st.write(Distace_from_capital)
Phone_number = st.text_input('Enter Phone Number',0)
Phone_category = devidePhoneNumber(Phone_number)
st.write(Phone_category)
Dishlikes = st.text_area('seprate by "," for more than one dish ')
NumOfDishlikes = dishlikesnumber(Dishlikes)
st.write(NumOfDishlikes)
Online = st.selectbox('is it provide online order' , ['Yes','No'],index=0)
print(type(Online))
orderonline = OnlineOrder(Online)
print(orderonline)
# st.write(orderonline)
table = st.selectbox('is it prefer book table' , ['Yes','No'])
booktable = BookTable(table)
print(booktable)
# st.write(booktable)

NameOfRestaurant = st.text_input('Name Of Restaurant')
Restaurant_type = st.text_input('Restaurant type') 
Restaurant_Cuisins = st.text_input('Restaurant cisins')
Restaurant_listedin_type = st.text_input('Restaurant Listed In Type') 

EncodedValues = [NameOfRestaurant,Restaurant_type,Restaurant_Cuisins,Restaurant_listedin_type,Restaurant_City]
Encoder = LabelEncoder()
Encoder.fit(EncodedValues)
value = Encoder.transform(EncodedValues)
value = value.tolist()
final_data = [votes, approx_cost_For_two_people,Distace_from_capital,Phone_category,NumOfDishlikes] + value + orderonline + booktable 

final_data_array=np.array(final_data)
print(final_data)


modelSuccess = model.predict([final_data_array])

if modelSuccess == 0 :
    SuccessorNot =  ' This Restaurant Not to be success'
elif modelSuccess == 1 : 
    SuccessorNot =  ' This Restaurant will be success'
if st.button("Predict"):
    st.success(f"Your predicted Success is {SuccessorNot}")



