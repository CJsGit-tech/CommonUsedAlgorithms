import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix

st.header("Machine Learning App")
st.sidebar.header("Commonly Used Algorithms")

try:
    file = st.file_uploader("Choose a CSV File",type = ['csv'])
    df = pd.read_csv(file)
    cols = df.columns
    num_cols = len(cols)
    num_rows = len(df)
    st.dataframe(df.head())
    st.write("Number of Columns: ",num_cols)
    st.write("Number of records",num_rows)
 
    st.markdown("Missing Values?")
    st.write(df.isnull().sum())
    st.markdown("Basic Stats")
    st.write(df.describe())
except:
    st.error("Please Input a Dataset")

st.write("---")

st.subheader("Exploratory Data Analysis")
x = st.radio("Show Pairplot?",["No","Yes"])
st.warning("Not Recomended when dataset has too many columns")
if x == "No":
    pass
elif x =="Yes":
    st.markdown("Correlation")
    sns.pairplot(df)
    plt.tight_layout()
    st.pyplot()

st.markdown("Jointplot")
try:
    col_1 = 0
    col_2 = 0
    col_1,col_2  = st.multiselect("Choose 2 Desired Columns to plot",[x for x in cols])
    sns.jointplot(x = col_1, y = col_2,data = df)
    plt.tight_layout()
    st.pyplot()
except:
    st.error("Selected Columns are invalid or are not iterable, or selcted columns are greater than 2")
    pass
st.write("---")

st.header("Train Test Split")

from sklearn.model_selection import train_test_split
st.subheader("Decide Your Independent Variables")
independents = st.multiselect("Choose Desired Columns",[col for col in cols])
st.markdown("You Have Selceted {}".format(independents))
x = df[[ item for item in independents]]
st.subheader("Decide Your Depende Variable")
y = df[st.selectbox("Choose a Column",[col for col in cols])]

st.markdown("Set Train Test Split Parameters")
test_size = st.number_input("test_size",min_value = 0.1,value = 0.3,max_value = 1.0,step = 0.1)


stratify = st.radio("Enable stratify fashion",["None","Yes","No"])

if stratify =="Yes":
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = test_size,stratify = y,random_state =101)
    st.info("Stratified Fashion Enabled")
elif stratify =="No":
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = test_size,stratify = y,random_state =101)
    st.info("Stratified Fashion Disabled")

########################################Logistic Regression#################################
st.write("---")
if st.sidebar.checkbox("LogisticRegression"):
    st.header("Logistic Regression")
    from sklearn.linear_model import LogisticRegression
    try:
        logistic = LogisticRegression()
        logistic.fit(x_train,y_train)
        st.success("Data Fit Successully")
    except:
        st.warning("Data Fitting Failed")
        pass

    st.subheader("Predictions and Evaluations")
    st.markdown("True:y_test VS Predictions")
    preds_log = logistic.predict(x_test)
    st.markdown("classification_report")
    st.code(classification_report(y_test,preds_log))
    st.markdown("Confusion Matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test,preds_log),index = ["No","Yes"],columns = ["No","Yes"]))

########################################Logistic Regression#################################

########################################K Nearest Neighbors#################################

if st.sidebar.checkbox("K-Nearest Neighbors"):
    st.header("K-Nearest Neighbors")
    from sklearn.neighbors import KNeighborsClassifier
    try:
        knn = KNeighborsClassifier(n_neighbors=st.number_input("input k",value = 5,min_value = 2))
        knn.fit(x_train,y_train)
        st.success("Data Fit Successully")
    except:
        st.warning("Data Fitting Failed")
        pass

    st.subheader("Predictions and Evaluations")
    st.markdown("True:y_test VS Predictions")
    preds_knn = knn.predict(x_test)
    st.markdown("classification_report")
    st.code(classification_report(y_test,preds_knn))
    st.markdown("Confusion Matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test,preds_knn),index = ["No","Yes"],columns = ["No","Yes"]))

    errors = st.radio("Show Error Rates:k until 40",["Off","On"])
    if errors == "Off":
        pass
    elif errors == "On":
        error_rate = []
        # Run multiple simulations
        for n in range(1,40):
            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(x_train,y_train)
            pred_i = knn.predict(x_test)
            error_rate.append(np.mean(pred_i != y_test))
        st.subheader("Plot Error Rates")
        plt.plot(range(1,40),error_rate, linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        st.pyplot()
########################################K Nearest Neighbors#################################
########################################Decision Tree Classifier#################################
if st.sidebar.checkbox("Decision Tree"):
    st.header("Decision Tree Classifier")
    from sklearn.tree import DecisionTreeClassifier
    try:
        tree = DecisionTreeClassifier()
        tree.fit(x_train,y_train)
        st.success("Data Fit Successully")
    except:
        st.warning("Data Fitting Failed")
        pass

    st.subheader("Predictions and Evaluations")
    st.markdown("True:y_test VS Predictions")
    preds_tree = tree.predict(x_test)
    st.markdown("classification_report")
    st.code(classification_report(y_test,preds_tree))
    st.markdown("Confusion Matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test,preds_tree),index = ["No","Yes"],columns = ["No","Yes"]))

if st.sidebar.checkbox("Random Forest Classifier"):
    st.header("Random Forest Classifier")
    from sklearn.ensemble import RandomForestClassifier
    try:
        random = RandomForestClassifier(n_estimators = 100,random_state=101)
        random.fit(x_train,y_train)
        st.success("Data Fit Successully")
    except:
        st.warning("Data Fitting Failed")
        pass

    st.subheader("Predictions and Evaluations")
    st.markdown("True:y_test VS Predictions")
    preds_ran = random.predict(x_test)
    st.markdown("classification_report")
    st.code(classification_report(y_test,preds_ran))
    st.markdown("Confusion Matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test,preds_ran),index = ["No","Yes"],columns = ["No","Yes"]))

########################################Decision Tree Classifier#################################

########################################Support Vector Machine#################################
if st.sidebar.checkbox("Support Vector Machine"):
    st.header("Support Vector Machine")
    from sklearn.svm import SVC
    try:
        svm = SVC()
        svm.fit(x_train,y_train)
        st.success("Data Fit Successfully")
    except:
        st.warning("Data Fitting Failed")
        pass


    st.subheader("Predictions and Evaluations")
    st.markdown("True:y_test VS Predictions")
    preds_svm = svm.predict(x_test)
    st.markdown("classification_report")
    st.code(classification_report(y_test,preds_svm))
    st.markdown("Confusion Matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test,preds_svm),index = ["No","Yes"],columns = ["No","Yes"]))

    st.write("---")
    st.subheader("GridSearch for the best potential Parameters")
    if st.checkbox("GridSearch"):
        from sklearn.model_selection import GridSearchCV
        param = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
        pipeline = GridSearchCV(SVC(),param,refit=True,verbose=2)
        grid = st.code(pipeline)
        pipeline.fit(x_train,y_train)

        st.subheader("Predictions and Evaluations")
        st.markdown("True:y_test VS Predictions")
        grid_pred = pipeline.predict(x_test)
        st.markdown("classification_report")
        st.code(classification_report(y_test,grid_pred))
        st.markdown("Confusion Matrix")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test,grid_pred),index = ["No","Yes"],columns = ["No","Yes"]))

######################################## Support Vector Machine #################################

######################################## K-means Clustering #################################
if st.sidebar.checkbox("K-means Clustering"):
    st.header("K-Means Clustering")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    pred_kmeans = kmeans.fit(x_train)
    st.code(pred_kmeans)


    st.subheader("Predictions and Evaluations")
    st.markdown("True:y_test VS Predictions")
    st.markdown("classification_report")
    st.code(classification_report(y_train,kmeans.labels_))
    st.markdown("Confusion Matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y_train,kmeans.labels_),index = ["No","Yes"],columns = ["No","Yes"]))

######################################## K-means Clustering #################################
