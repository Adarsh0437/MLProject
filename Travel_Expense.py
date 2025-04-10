import streamlit as st
import pandas as pd
import pickle
import base64

# Load model and preprocessor
with open("Travel_Expensemodel.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load datasets
try:
    flights_df = pd.read_csv("flights.csv")
    hotels_df = pd.read_csv("hotels.csv")

    # Clean column names
    flights_df.columns = flights_df.columns.str.strip().str.lower()
    hotels_df.columns = hotels_df.columns.str.strip().str.lower()

    # Extract first name from locations (before " (")
    from_locations = sorted(set([loc.split(" (")[0] for loc in flights_df['from'].dropna()])) \
        if 'from' in flights_df.columns else []

    to_locations = sorted(set([loc.split(" (")[0] for loc in flights_df['to'].dropna()])) \
        if 'to' in flights_df.columns else []

    places = sorted(set([place.split(" (")[0] for place in hotels_df['place'].dropna()])) \
        if 'place' in hotels_df.columns else []

    # Add default placeholders
    from_locations = ["Select a location"] + from_locations
    to_locations = ["Select a location"] + to_locations
    places = ["Select a place"] + places

except Exception as e:
    st.error(f"❌ Error loading CSV files: {e}")
    from_locations, to_locations, places = ["Select a location"], ["Select a location"], ["Select a place"]

# Flight type mapping
flight_type_display = {
    "Economy": 0,
    "Business": 1,
    "First Class": 2
}
flight_type_options = ["Select a flight type"] + list(flight_type_display.keys())

# Streamlit Page Config
st.set_page_config(page_title="Travel Expense Predictor", layout="centered")
st.title("🧳 Total Travel Expense Predictor")
st.markdown("Predict your estimated total **travel expense** based on your trip details.")
# Background image setup
def set_bg(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg("Travel.jpg")  # Make sure the image is in the same folder
# UI Layout
with st.form("expense_form"):
    col1, col2 = st.columns(2)

    with col1:
        hotel_price = st.number_input("🏨 Hotel Price (0-350 $)", min_value=0,max_value=350,step=10)
        from_location = st.selectbox("📍 From Location", from_locations, index=0)
        days = st.slider("📅 Number of Days(1-5)", min_value=1, max_value=5, value=1)
        flight_type_input = st.selectbox("✈️ Flight Type", flight_type_options)

    with col2:
        flight_price = st.number_input("💺 Flight Price (0-2000 $)", min_value=0,max_value=2000,step=50)
        to_location = st.selectbox("📍 To Location", to_locations, index=0)
        distance = st.slider("🛣️ Distance (100-1000 KM)", min_value=100,max_value=1000)
        place = st.selectbox("🏙️ Place of Stay", places, index=0)

    submitted = st.form_submit_button("🚀 Predict Total Expense")

    if submitted:
        # Validation check
        if from_location == "Select a location" or to_location == "Select a location" or place == "Select a place":
            st.warning("⚠️ Please select valid From, To, and Place of Stay locations.")
        else:
            try:
                flightType = flight_type_display[flight_type_input]
                total_hotel_price = hotel_price * days

                # Input DataFrame
                input_df = pd.DataFrame([{
                    'hotel_price': hotel_price,
                    'flight_price': flight_price,
                    'from_location': from_location,
                    'days': days,
                    'to_location': to_location,
                    'flightType': flightType,
                    'distance': distance,
                    'place': place,
                    'total_hotel_price': total_hotel_price
                }])

                # Transform and predict
                transformed_input = preprocessor.transform(input_df)
                prediction = model.predict(transformed_input)[0]

                st.success(f"💰 Estimated Total Travel Expense: $ {prediction:,.2f}")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
