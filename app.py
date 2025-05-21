import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# load the diabetes dataset
diabetes_df = pd.read_csv('diabetes.csv')

# group the data by outcome to get a sense of the distribution
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# split the data into input and target variables
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# scale the input variables using StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create an SVM model with a linear kernel
model = svm.SVC(kernel='linear')

# train the model on the training set
model.fit(X_train, y_train)

# make predictions on the training and testing sets
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

# calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)

# create the Streamlit app
def app():

    img = Image.open(r"img.jpeg")
    img = img.resize((200,200))
    st.image(img,caption="Diabetes Image",width=200)


    st.title('Diabetes Prediction')

    # create the input form for the user to input new data
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # make a prediction based on the user input
    # input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    # input_data_nparray = np.asarray(input_data)
    # reshaped_input_data = input_data_nparray.reshape(1, -1)
    # prediction = model.predict(reshaped_input_data)
#     input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
#     input_data_nparray = np.asarray(input_data).reshape(1, -1)

# # ✅ Apply the same scaler used during training
#     scaled_input = scaler.transform(input_data_nparray)

# # Use scaled input for prediction
#     prediction = model.predict(scaled_input)
    input_data = {
    'Pregnancies': [preg],
    'Glucose': [glucose],
    'BloodPressure': [bp],
    'SkinThickness': [skinthickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
      }

    input_df = pd.DataFrame(input_data)

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)


    # display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if prediction == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    # display some summary statistics about the dataset
    st.header('Dataset Summary')
    st.write(diabetes_df.describe())

    st.header('Distribution by Outcome')
    st.write(diabetes_mean_df)

    # display the model accuracy
    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {train_acc:.2f}')
    st.write(f'Test set accuracy: {test_acc:.2f}')

if __name__ == '__main__':
    app()

# import numpy as np
# import pandas as pd
# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from PIL import Image
# import matplotlib.pyplot as plt
# import seaborn as sns

# # load the diabetes dataset
# diabetes_df = pd.read_csv('diabetes.csv')

# # group the data by outcome to get a sense of the distribution
# diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# # split the data into input and target variables
# X = diabetes_df.drop('Outcome', axis=1)
# y = diabetes_df['Outcome']

# # scale the input variables using StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)

# # split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# # create an SVM model with a linear kernel
# model = svm.SVC(kernel='linear')
# model.fit(X_train, y_train)

# # predictions and accuracy
# train_y_pred = model.predict(X_train)
# test_y_pred = model.predict(X_test)
# train_acc = accuracy_score(train_y_pred, y_train)
# test_acc = accuracy_score(test_y_pred, y_test)

# # Streamlit app
# def app():
#     img = Image.open("img.jpeg")
#     img = img.resize((200,200))
#     st.image(img, caption="Diabetes Image", width=200)

#     st.title('Diabetes Prediction')

#     # Sidebar for inputs
#     st.sidebar.title('Input Features')
#     preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
#     glucose = st.sidebar.slider('Glucose', 0, 199, 117)
#     bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
#     skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
#     insulin = st.sidebar.slider('Insulin', 0, 846, 30)
#     bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
#     dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
#     age = st.sidebar.slider('Age', 21, 81, 29)

#     # Prepare input for prediction
#     input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
#     input_data_nparray = np.asarray(input_data).reshape(1, -1)
#     scaled_input = scaler.transform(input_data_nparray)
#     prediction = model.predict(scaled_input)

#     # Display prediction
#     st.write('Based on the input features, the model predicts:')
#     if prediction == 1:
#         st.warning('This person has diabetes.')
#     else:
#         st.success('This person does not have diabetes.')

#     # Dataset Summary
#     st.header('Dataset Summary')
#     st.write(diabetes_df.describe())

#     # Distribution by Outcome
#     st.header('Distribution by Outcome')
#     st.write(diabetes_mean_df)

#     # Model Accuracy
#     st.header('Model Accuracy')
#     st.write(f'Train set accuracy: {train_acc:.2f}')
#     st.write(f'Test set accuracy: {test_acc:.2f}')

#     # Correlation Heatmap
#     st.header('Feature Correlation Heatmap')
#     corr_matrix = diabetes_df.corr()
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#     st.pyplot(plt.gcf())
#     plt.clf()

#     # SVM Feature Importance
#     st.header("SVM Model Feature Coefficients")
#     feature_names = diabetes_df.columns[:-1]
#     coefficients = model.coef_[0]
#     importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
#     st.dataframe(importance_df.sort_values(by="Coefficient", ascending=False))

# if __name__ == '__main__':
#     app()

