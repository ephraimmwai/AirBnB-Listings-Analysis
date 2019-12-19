import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


train_df = pd.read_csv('data/predict_price.csv')
int_cols = np.setdiff1d(train_df.columns.tolist(), train_df.select_dtypes(include=['object']).columns.tolist())


#Define Tukey method to handle outliers

def tukey_rule(df, col):
    data = df[col]
    Q1 = data.quantile(.25)
    Q3 = data.quantile(.75)
    IQR = Q3-Q1
    max_val = Q3+1.5*IQR
    min_val = Q1-1.5*IQR
    
    return df[(df[col] > min_val)&(df[col]< max_val)]

#define a funtion to create dummy columns
def create_dummy_cols(df, cat_cols, dummy_na):
    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df      


df_outlier_removed = train_df.copy()
#integer columns we need to remove outliers -- scatter plot indicates outliers
int_cols = ['bathrooms','accommodates', 'bedrooms', 'beds','extra_people',
        'host_listings_count', 'maximum_nights',
       'minimum_nights', 'price']

for col in int_cols:
    df_outlier_removed = tukey_rule(df_outlier_removed, col)


train_df = df_outlier_removed.dropna()


#Drop rows where the price has missing values
train_df  = train_df.dropna(subset=['price'], axis=0)
#create a list of the column names of categorical variables
cat_df = train_df.select_dtypes(include=['object'])
cat_cols_lst = cat_df.columns

#create dummy columns
train_df = create_dummy_cols(train_df, cat_cols_lst, dummy_na=False)

# new_train_df.columns.tolist()

nulls_df = train_df.isnull().sum().reset_index().rename(columns={'index':'col_name', 0:'nulls'})
nulls_df = nulls_df[nulls_df['nulls']>0]
#Check for any null values

new_nulls_df = train_df.isnull().sum().reset_index().rename(columns={'index':'col_name', 0:'nulls'})
new_nulls_df = new_nulls_df[new_nulls_df['nulls']>0]

X_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds','cancellation_policy_moderate', 'cancellation_policy_strict','cancellation_policy_strict_14_with_grace_period','cancellation_policy_super_strict_30','cancellation_policy_super_strict_60', 'extra_people','guests_included', 'host_listings_count', 'instant_bookable_t','maximum_nights', 'minimum_nights','require_guest_phone_verification_t', 'require_guest_profile_picture_t','room_type_Private room', 'room_type_Shared room']

#Split into explanatory and response variables
X = train_df.drop('price', axis=1).reindex(columns=sorted(X_cols))
y = train_df['price']

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42) 

print('Creating Price Prediction Model...')

lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit
        
#Predict and score the model
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train) 

test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)

print("Training data score: {}.\nTest data score {}.".format(train_score, test_score))


import pickle
pickle.dump(lm_model, open("price_model.pkl","wb"))

print('Model saved successfully!')


