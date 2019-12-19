import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_price',methods=['POST'])
def predict_price():

    cols = ['accommodates','bathrooms','bedrooms','beds','cancellation_policy', 'extra_people','guests_included','host_listings_count', 'instant_bookable', 'maximum_nights', 'minimum_nights', 'require_guest_phone_verification','room_type']

    X_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
           'cancellation_policy_moderate', 'cancellation_policy_strict',
           'cancellation_policy_strict_14_with_grace_period',
           'cancellation_policy_super_strict_30',
           'cancellation_policy_super_strict_60', 'extra_people',
           'guests_included', 'host_listings_count', 'instant_bookable_t',
           'maximum_nights', 'minimum_nights',
           'require_guest_phone_verification_t', 'require_guest_profile_picture_t',
           'room_type_Private room', 'room_type_Shared room']

    features_data = [x for x in request.form.values()]

    listing_details = {}

    for i,v in zip(cols,features_data):
        listing_details[i] = v

    df =pd.DataFrame(data = listing_details, index=[0])


    df[['accommodates','bathrooms','bedrooms','beds','extra_people','guests_included','host_listings_count','maximum_nights','minimum_nights']] = df[['accommodates','bathrooms','bedrooms','beds','extra_people','guests_included','host_listings_count','maximum_nights','minimum_nights']].astype(int)


    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    int_cols = df.select_dtypes(include=['int']).columns.tolist()


    dummy = pd.get_dummies(df[cat_cols])

    df = pd.concat([df.drop(cat_cols, axis=1), dummy], axis=1)
    df = df.reindex(columns=sorted(df.columns))

    dc = df.to_dict('records')
    zero_dic = {}

    cls_dc = list(dc[0].keys())
    diff_cols = set(X_cols) - set(cls_dc)
    #append a 0 value
    for d in diff_cols:
        zero_dic[d]= 0

    feature_dict = {**dc[0],**zero_dic}
    feature = {k:v for k,v in feature_dict.items() if k in X_cols}

    values = list(feature.values())
    price_pred = int(round(model.predict([values])[0],-1))

    return render_template('index.html', prediction_text= '$ {}'.format(price_pred))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)