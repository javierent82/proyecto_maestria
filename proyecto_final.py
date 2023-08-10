#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_samples
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer



#%%
### DATA MERGE AND FACTS TABLE BUILDING

# Load the data from CSV files into DataFrames
customer_df = pd.read_csv('customer.csv')
device_df = pd.read_csv('device.csv')
page_views_df = pd.read_csv('page_views.csv')
subscriptions_df = pd.read_csv('subscriptions.csv')
purchases_df = pd.read_csv('purchase_v2.csv')

# Calculate the number of devices purchased as the count of unique purchases per customer
num_devices_purchased = purchases_df.groupby('clientId')['deviceId'].nunique().reset_index(name='num_devices_purchased')

# Calculate the average time between purchases per clientId
purchases_df['purchase_dt'] = pd.to_datetime(purchases_df['purchase_dt'])
purchases_df.sort_values(['clientId', 'purchase_dt'], inplace=True)
purchases_df['time_between_purchases'] = purchases_df.groupby('clientId')['purchase_dt'].diff().dt.days
average_time_between_purchases = purchases_df.groupby('clientId')['time_between_purchases'].mean().reset_index(name='avg_time_between_purchases')

# Find the first purchase for each custid
first_purchase_date = purchases_df.groupby('clientId')['purchase_dt'].min().reset_index(name='first_purchase_date')

# Filter subscriptions to only include records with subscription_status = 1
filtered_subscriptions_df = subscriptions_df[subscriptions_df['subscription_status'] == 1]

# Find the latest subscription with subscription_status = 1 for each clientId
latest_subscription_date = filtered_subscriptions_df.groupby('custid')['subscription_date'].max().reset_index(name='latest_subscription_date')
latest_subscription_status = filtered_subscriptions_df.groupby('custid')['subscription_status'].last().reset_index(name='subscription_status')

# Calculate the average page views per month per custId
page_views_df['event_date'] = pd.to_datetime(page_views_df['event_date'])
page_views_df['month'] = page_views_df['event_date'].dt.to_period('M')
avg_page_views = page_views_df.groupby(['custId', 'month']).size().groupby('custId').mean().reset_index(name='avg_page_views_per_month')

# Merge all the calculated fields with the customer dataset
facts_df = pd.merge(customer_df, num_devices_purchased, left_on='custid', right_on='clientId', how='left')
facts_df = pd.merge(facts_df, average_time_between_purchases, left_on='custid', right_on='clientId', how='left')
facts_df = pd.merge(facts_df, latest_subscription_date, on='custid', how='left')
facts_df = pd.merge(facts_df, latest_subscription_status, on='custid', how='left')
facts_df = pd.merge(facts_df, first_purchase_date, left_on='custid', right_on='clientId', how='left')
facts_df = pd.merge(facts_df, avg_page_views, left_on='custid', right_on='custId', how='left')

# Convert date columns to datetime type
facts_df['latest_subscription_date'] = pd.to_datetime(facts_df['latest_subscription_date'])
facts_df['first_purchase_date'] = pd.to_datetime(facts_df['first_purchase_date']).dt.tz_localize(None)

# Calculate the time to first subscription as the difference between first purchase and latest subscription
facts_df['time_to_first_subscription'] = (facts_df['latest_subscription_date'] - facts_df['first_purchase_date']).dt.days

# Select and rename the required fields
facts_df = facts_df[['custid', 'num_devices_purchased', 'avg_time_between_purchases',
                     'subscription_status', 'time_to_first_subscription',
                     'avg_page_views_per_month', 'gender', 'zip_code',
                     'household_type', 'income_level']]

# Save the facts dataset to a CSV file
facts_df.to_csv('facts_dataset.csv', index=False)

# %%
### DATA CLEAN UP
facts_df.drop(['custid'], axis=1, inplace=True)

# Convert categorical columns to numeric scale using LabelEncoder
categorical_cols = ['gender', 'household_type', 'income_level']
label_encoder = LabelEncoder()

for col in categorical_cols:
    facts_df[col] = label_encoder.fit_transform(facts_df[col])

# Normalize numerical columns using Min-Max scaling
numerical_cols = ['num_devices_purchased', 'avg_time_between_purchases',
                  'subscription_status', 'time_to_first_subscription',
                  'avg_page_views_per_month', 'zip_code']

scaler = MinMaxScaler()
facts_df[numerical_cols] = scaler.fit_transform(facts_df[numerical_cols])

# Save the cleaned and normalized dataset
facts_df.to_csv('cleaned_normalized_facts_df.csv', index=False)


# %%
### EDA

# Display summary statistics
summary_stats = facts_df.describe()
print("Summary Statistics:\n", summary_stats)

# Correlation matrix
corr_matrix = facts_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Histograms of numerical features
numerical_cols = ['num_devices_purchased', 'avg_time_between_purchases',
                  'subscription_status', 'time_to_first_subscription',
                  'avg_page_views_per_month', 'zip_code']

facts_df[numerical_cols].hist(figsize=(10, 8))
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

# Bar plots of categorical features
categorical_cols = ['gender', 'household_type', 'income_level']

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=facts_df)
    plt.title(f'Count of Customers by {col}')
    plt.show()

# Box plots of numerical features by categorical variables
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y='num_devices_purchased', data=facts_df)
    plt.title(f'Box Plot of num_devices_purchased by {col}')
    plt.show()


# %%
### SEGMENTATION
# Select the features for segmentation
features = ['num_devices_purchased', 'avg_time_between_purchases',
            'subscription_status', 'time_to_first_subscription',
            'avg_page_views_per_month', 'gender', 'zip_code',
            'household_type', 'income_level']

# Handle NaN values with mean imputation
imputer = SimpleImputer(strategy='mean')
facts_df[features] = imputer.fit_transform(facts_df[features])

# Create a K-means object
kmeans = KMeans(random_state=42)

# Define the parameter grid for grid search
param_grid = {'n_clusters': [3, 4, 5, 6, 7]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(kmeans, param_grid, cv=5)
grid_search.fit(facts_df[features])

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best clustering model
best_model = grid_search.best_estimator_

# Assign cluster labels to the dataset
facts_df['cluster'] = best_model.predict(facts_df[features])

# Evaluate the segmentation using silhouette score
silhouette_avg = silhouette_score(facts_df[features], facts_df['cluster'])
print("Silhouette Score:", silhouette_avg)

# View the cluster distribution
cluster_counts = facts_df['cluster'].value_counts().sort_index()
print("Cluster Distribution:\n", cluster_counts)

# View the cluster centroids
cluster_centroids = pd.DataFrame(best_model.cluster_centers_, columns=features)
print("Cluster Centroids:\n", cluster_centroids)


# %%
### PLOT RESULTS
# Plot the cluster distribution
cluster_counts = pd.Series(y_val_pred).value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Cluster Distribution')
plt.show()

# Calculate silhouette scores for each sample
silhouette_samples = silhouette_samples(X_val, y_val_pred)

# Plot the silhouette scores
plt.hist(silhouette_samples, bins=np.arange(-1, 1.1, 0.1))
plt.xlabel('Silhouette Score')
plt.ylabel('Frequency')
plt.title('Silhouette Scores')
plt.show()
# %%

## PLOT 2

# Scatter Plot
plt.figure(figsize=(10, 8))
plt.scatter(facts_df['avg_page_views_per_month'], facts_df['num_devices_purchased'], c=facts_df['cluster'], cmap='viridis')
plt.xlabel('Average Page Views per Month')
plt.ylabel('Number of Devices Purchased')
plt.title('Scatter Plot of Average Page Views vs Number of Devices Purchased with Clusters')
plt.show()

# Bar Plot
plt.figure(figsize=(10, 6))
cluster_counts = facts_df['cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Cluster Distribution')
plt.show()

# Radar Plot
cluster_centroids_scaled = scaler.inverse_transform(cluster_centroids[numerical_cols])  # Scale back to original values
cluster_labels = ['Cluster ' + str(i) for i in range(len(cluster_centroids))]
num_features = len(numerical_cols)

angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
angles += angles[:1]  # Include the first angle to close the plot

plt.figure(figsize=(8, 8))
for i in range(len(cluster_centroids_scaled)):
    values = cluster_centroids_scaled[i].tolist()
    values += values[:1]  # Include the first value to close the plot
    plt.polar(angles, values, label=cluster_labels[i])
plt.xticks(angles[:-1], numerical_cols)
plt.yticks([])
plt.title('Radar Plot of Cluster Centroids')
plt.legend()
plt.show()
# %%
