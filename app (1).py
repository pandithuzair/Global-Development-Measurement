import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import seaborn as sns

# Load the trained model (Pickle)
with open('kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

# Title of the app
st.title("World Development Measurement Clustering Analysis")

# Input fields for the user to enter data
Birth_Rate = st.number_input("Enter Birth Rate")
Business_Tax_Rate = st.number_input("Enter Business Tax Rate")
CO2_Emissions = st.number_input("Enter CO2 Emissions")
Days_to_start_business = st.number_input("Enter Days to start business")
Enery_usage = st.number_input("Enter Energy usage")
Health_Exp_GDP = st.number_input("Enter Health Exp GDP")
Hours_to_do_tax = st.number_input("Enter Hours to do tax")
Infant_Mortality_Rate = st.number_input("Enter Infant Mortality Rate")
Internet_Usage = st.number_input("Enter Internet Usage")
Lending_Interest = st.number_input("Enter Lending Interest")
Life_Expectancy_Female = st.number_input("Enter Life Expectancy Female")
Life_Expectancy_Male = st.number_input("Enter Life Expectancy Male")
Mobile_Phone_Usage = st.number_input("Enter Mobile Phone Usage")
Population_0_14 = st.number_input("Enter Population 0-14")
Population_15_64 = st.number_input("Enter Population 15-64")
Population_65 = st.number_input("Enter Population 65+")
Population_Total = st.number_input("Enter Population Total")
Population_Urban = st.number_input("Enter Population Urban")

# Create a submit button
submit_button = st.button("Submit")

# When the button is clicked, make a prediction
if submit_button:
    # Prepare the input data as a NumPy array
    input_data = np.array([[Birth_Rate, Business_Tax_Rate, CO2_Emissions, Days_to_start_business, Enery_usage, Health_Exp_GDP, 
                            Hours_to_do_tax, Infant_Mortality_Rate, Internet_Usage, Lending_Interest, Life_Expectancy_Female, 
                            Life_Expectancy_Male, Mobile_Phone_Usage, Population_0_14, Population_15_64, Population_65, 
                            Population_Total, Population_Urban]])

    # Predict the cluster using the trained KMeans model
    cluster_label = kmeans.predict(input_data)
    predicted_cluster = cluster_label[0]  # Get the predicted cluster label

    # Display the predicted cluster
    st.write(f"The data belongs to Cluster {predicted_cluster}")  # Display the predicted cluster as the data belongs to that cluster

    # Get cluster centroids
    centroids = kmeans.cluster_centers_

    # Calculate the distances from input data to each centroid
    distances = cdist(input_data, centroids, 'euclidean')

    # Display the distances
    st.subheader("Distances to Each Cluster Centroid")
    for i, distance in enumerate(distances[0]):
        st.write(f"Distance to Cluster {i} Centroid: {distance:.2f}")

    st.subheader("Cluster Centroids")
    for i, centroid in enumerate(centroids):
        st.write(f"Centroid {i}: {centroid}")

    # Plotting the distances as a bar plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar plot for the distances from the input data point to each centroid
    bars = ax.bar(range(len(distances[0])), distances[0], color='skyblue', label='Distance to Centroids')

    # Adding labels and title
    ax.set_xticks(range(len(distances[0])))
    ax.set_xticklabels([f"Cluster {i}" for i in range(len(distances[0]))])
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Distance to Centroid')
    ax.set_title('Distances from Input Data Point to Cluster Centroids')

    # Adding legend
    ax.legend(title="Legend", loc="upper right", labels=["Distance to Centroids"])

    # Display the distances on top of the bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()  # Get the height of the bar (i.e., the distance)
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)

    # Display the plot
    st.pyplot(fig)

    # Highlighting the closest cluster
    closest_cluster = np.argmin(distances)  # Find the index of the closest centroid
    st.write(f"The closest cluster is Cluster {closest_cluster}, with a distance of {distances[0][closest_cluster]:.2f}")
