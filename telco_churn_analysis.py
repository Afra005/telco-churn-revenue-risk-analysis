import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
#step1
print(df.head()) #shows first 5 rows
print(df.shape) #rows, cols
print(df.columns)#shows all the col name
df.info()#each col,non-nulls,dtypes,memory
#step2
print(df.describe())
print(df.describe(include="object"))
print("=== Churn Value Counts ===")
print(df["Churn"].value_counts())
#step3
churn_rate = df["Churn"].value_counts(normalize=True)*100
print("=== Churn Rate (%) ===")
print(churn_rate)
print("=== Total Monthly Charges by Churn ===")
print(df.groupby("Churn")["MonthlyCharges"].sum())
rev_by_churn = df.groupby("Churn")["MonthlyCharges"].sum()
revenue_at_risk = (rev_by_churn["Yes"] / rev_by_churn.sum()) * 100
print(f"Revenue at Risk (% from churned customers): {revenue_at_risk:.2f}%")
#what percentage of the rev comes from
# the customer who churned yes
print("=== Contract Type Counts ===")
print(df["Contract"].value_counts())
print("=== Churn Rate by Contract Type (%) ===")
print(df.groupby("Contract")["Churn"].value_counts(normalize=True)*100)
#month-to-month customers have the highest observed churn(=yes) rate
#Do high monthly charges predict churn?
print("=== Average Monthly Charges by Churn ===")
print(df.groupby("Churn")["MonthlyCharges"].mean())
#Customers (Churn = Yes) had a higher charge than customers who stayed (Churn = No)
#churn = yes all 3
print("=== Churn Rate by Tenure (%) ===")
print(df.groupby("tenure")["Churn"].value_counts(normalize=True)*100)
#Customers with shorter tenure are substantially more likely to churn=yes than long-term customers.
print("=== Churn Rate by Internet Service (%) ===")
print(df.groupby("InternetService")["Churn"].value_counts(normalize=True)*100)
#Fiber optic have the highest observed churn=yes
print("=== Churn Rate by Payment Method (%) ===")
print(df.groupby("PaymentMethod")["Churn"].value_counts(normalize=True)*100)
#electronic check have the highest observed churn=yes rate
#step4
#x = customer information y = Churn
#TotalCharges should be on the numeric list but its in the categorical
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"TotalCharges dtype: {df['TotalCharges'].dtype}")
print(f"TotalCharges nulls before fill: {df['TotalCharges'].isnull().sum()}")
df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].median())
print(f"TotalCharges nulls after fill: {df['TotalCharges'].isnull().sum()}")
x = df.drop(["customerID", "Churn"], axis = 1)
y = df["Churn"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12,stratify=y)
#test size 20%, to split at a selected random number for 20% , to keep the proportions same while splitting
categorical_cols = x_train.select_dtypes(include="object").columns
numerical_cols = x_train.select_dtypes(exclude="object").columns
print("=== Categorical Columns ===")
print(categorical_cols)
print("=== Numerical Columns ===")
print(numerical_cols)
preprocessor = ColumnTransformer([("categorical",OneHotEncoder(),categorical_cols),
                ("numerical", StandardScaler(),numerical_cols)])
#OneHotEncoder for encoding the non-numeric nums
#StandardScaler for encoding the numeric nums
x_train_processed = preprocessor.fit_transform(x_train)
#learns the rules of the dataset
x_test_processed = preprocessor.transform(x_test)
#applies learnt rules on the test split
print(f"X_train processed shape: {x_train_processed.shape}")
print(f"X_test processed shape: {x_test_processed.shape}")
#data processed
#M1 Logistic Regression
model1 = LogisticRegression()
model1.fit(x_train_processed, y_train)
#learns the relationship between the customer information (X) and churn (y)
y_pred = model1.predict(x_test_processed)
#the customer information the model has not trained on predicts the churn answers
accuracy = accuracy_score(y_test,y_pred) #y_train was already used to teach the model
#compare test result and model predicted result
print(f"Accuracy: {accuracy*100:.2f}%")
confusion = confusion_matrix(y_test,y_pred)
#confusion_matrix walks through every testcustomer and compares actual ans vs model's pred ans
print("=== Confusion Matrix: Logistic Regression ===")
print(confusion)
print("=== Classification Report: Logistic Regression ===")
print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred))
y_prob = model1.predict_proba(x_test_processed)[:,1]
#give me only col1 (yes) i'll use it for roc auc
auc = roc_auc_score(y_test, y_prob)
#ROC curve  measures classification by plotting True post rates
# against False positive rates at different threshold
#AUC overall graph using ROC auc measures model's ability to distinguish yes or no churn
print(f"Auc = {auc*100:.2f}%")
#m2 Decision Tree
model2 = DecisionTreeClassifier(random_state=12)
model2.fit(x_train_processed, y_train)
y_pred_2 = model2.predict(x_test_processed)
accuracy_2 = accuracy_score(y_test,y_pred_2)
print(f"Accuracy_2: {accuracy_2*100:.2f}%")
confusion_2 = confusion_matrix(y_test,y_pred_2)
print("=== Confusion Matrix: Decision Tree ===")
print(confusion_2)
print("=== Classification Report: Decision Tree ===")
print(classification_report(y_test,y_pred_2))
print(classification_report(y_test,y_pred_2))
y_prob_2 = model2.predict_proba(x_test_processed)[:,1]
auc_2 = roc_auc_score(y_test, y_prob_2)
print(f"Auc_2 = {auc_2*100:.2f}%")
#m3 Random Forest
model3 = RandomForestClassifier(random_state=12)
model3.fit(x_train_processed, y_train)
y_pred_3 = model3.predict(x_test_processed)
confusion_3 = confusion_matrix(y_test,y_pred_3)
print("=== Confusion Matrix: Random Forest ===")
print(confusion_3)
print("=== Classification Report: Random Forest ===")
print(classification_report(y_test,y_pred_3))
y_prob_3 = model3.predict_proba(x_test_processed)[:,1]
auc_3 = roc_auc_score(y_test, y_prob_3)
print(f"Auc_3 = {auc_3*100:.2f}%")
accuracy_3 = accuracy_score(y_test,y_pred_3)
print(f"Accuracy_3: {accuracy_3*100:.2f}%")
fpr_1, tpr_1, _ = roc_curve(y_test,y_prob, pos_label="Yes")
# "_" i dont need the 3rd val throw it away
fpr_2, tpr_2, _ = roc_curve(y_test,y_prob_2, pos_label="Yes")
fpr_3, tpr_3, _ = roc_curve(y_test,y_prob_3, pos_label="Yes")
plt.plot(fpr_1,tpr_1,label="Logistic Regression (AUC = 85.40%)")
plt.plot(fpr_2,tpr_2,label="Decision Tree (AUC = 66.62%)")
plt.plot(fpr_3,tpr_3,label="Random Forest (AUC = 83.54%)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("roc_curve_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
features_name = preprocessor.get_feature_names_out()
print("=== Feature Names (After Preprocessing) ===")
print(features_name)
importances = model3.feature_importances_
#How important was each processed feature when making predictions?
print("=== Raw Feature Importances ===")
print(importances)
importances_df = pd.DataFrame({"Feature": features_name, "Importances": importances})
importances_df= importances_df.sort_values(by="Importances", ascending=False)
#sort_values() rearranges it based on the importance
#ascending=False means largest → smallest.
print("=== Feature Importances (Sorted) ===")
print(importances_df)
top_features = importances_df.head(10)
#takes the first 10 rows and stores them in top_features
top_features=top_features.copy()
#makes an independent copy of those 10 rows.
top_features["Feature"]=(top_features["Feature"]
        .str.replace("numerical__","",regex=False)
        .str.replace("categorical__","", regex=False))
#find "numerical__" and replace it with nothing
#Treat "numerical__" and "categorical__" as literal text, not as a regular expression pattern.
top_features["Feature"] = top_features["Feature"].replace({
    "TotalCharges": "Total Charges",
    "MonthlyCharges": "Monthly Charges",
    "Contract_Month-to-month": "Month-to-Month Contract",
    "OnlineSecurity_No": "No Online Security",
    "InternetService_Fiber optic": "Fiber Optic Internet",
    "PaymentMethod_Electronic check": "Electronic Check",
    "TechSupport_No": "No Tech Support",
    "Contract_Two year": "Two-Year Contract",
    "SeniorCitizen": "Senior Citizen"
})
print("=== Top 10 Features ===")
print(top_features)
plt.barh(top_features["Feature"], top_features["Importances"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("top_10_features_importances.png", dpi=300, bbox_inches = "tight")
plt.show()
#plt.barh(...) → creates horizontal bars.
#top_features["Feature"] → y axis names of the 10 features.
#top_features["Importances"] → x axis their importance values.
#invert_yaxis() → puts the most important feature at the top instead of the bottom.
#confusion matrix heatmap
fig,axes = plt.subplots(1,3, figsize=(15,4))
#entire fig, creates multiple plotting areas at once 1 row 3 cols
sns.heatmap(confusion, annot=True,xticklabels=["No","Yes"],cmap="coolwarm",fmt="d",
            linewidths=1, linecolor='white',yticklabels=["No","Yes"],ax=axes[0])
#confusion → provides the numbers annot=True → displays those numbers inside the boxes
#axes[0] col 1 axes[2] col 2
sns.heatmap(confusion_2, annot=True,xticklabels=["No","Yes"],cmap="coolwarm",fmt="d",
            linewidths=1, linecolor='white',yticklabels=["No","Yes"],ax=axes[1])
sns.heatmap(confusion_3, annot=True,xticklabels=["No","Yes"],cmap="coolwarm",fmt="d",
            linewidths=1, linecolor='white',yticklabels=["No","Yes"],ax=axes[2])
axes[0].set_title("Logistic Regression")
axes[1].set_title("Decision Tree")
axes[2].set_title("Random Forest")
fig.supxlabel("Predicted")
fig.supylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches = "tight")
plt.show()
#model comparison
model_comparison=pd.DataFrame({"Model":["Logistic Regression","Decision Tree","Random Forest"],
        "Accuracy":[accuracy,accuracy_2,accuracy_3],"AUC":[auc,auc_2,auc_3]})
print("=== Model Comparison (Accuracy & AUC) ===")
print(model_comparison)
#churn risk by segment
df["TenureGroup"]=pd.cut(df["tenure"],bins=[-1,11,23,35,47,59,float("inf")],
    labels=["0-11 Months","12-23 Months","24-35 Months","36-47 Months","48-59 Months","60+ Months"])
#pd.cut() decides which range a val belongs to
#-1 because otherwise it wont include 0, inf=positive infinity
print("=== Customer Count by Tenure Group ===")
print(df["TenureGroup"].value_counts().sort_index())
#Count customers are in each TenureGroup, arrange the groups in their defined order
tenure_churn_rate=(df.groupby("TenureGroup",observed=True)["Churn"]
          .apply(lambda x:(x=="Yes").mean()*100))
#lambda is a short way of writing a function
#Take the group's Churn values (x) and calculate the percentage that are "Yes"
print("=== Churn Rate by Tenure Group (%) ===")
print(tenure_churn_rate)
plt.figure(figsize=(9,5))
plt.bar(tenure_churn_rate.index,tenure_churn_rate.values)
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=30)#rotates the x axis labels 30 degree
plt.tight_layout()
plt.savefig("churn_rate_by_tenure.png", dpi=300, bbox_inches="tight")
plt.show()
contract_tenure_churn=(df.groupby(["TenureGroup","Contract"],observed=True)["Churn"]
          .apply(lambda x:(x=="Yes").mean()*100).unstack())
#.unstack() turns those combinations into the side by side table above.
print("=== Churn Rate by Tenure Group & Contract (%) ===")
print(contract_tenure_churn)
segment_summary = (df.groupby(["TenureGroup","Contract"],observed=True)
            .agg(Customers=("customerID","count"), MonthlyRevenue=("MonthlyCharges","sum")))
#All customers + all monthly revenue for each segment(Tenure+Contract)
print("=== Segment Summary (Customers & Revenue) ===")
print(segment_summary)
segment = df[(df["TenureGroup"] == "0-11 Months") &
    (df["Contract"] == "Month-to-month") &
    (df["Churn"] == "Yes")]
#now looking only at the monthly charges of those customers selected in segment and sum them
print(f"Segment Revenue (sum of MonthlyCharges): {segment['MonthlyCharges'].sum():.2f}")
#Monthly revenue from the customers in the segment who churned
tot_churned_yes_rev=df[df["Churn"]=="Yes"]["MonthlyCharges"].sum()
print(f"Total Churned Revenue (all Yes): {tot_churned_yes_rev:.2f}")
#Monthly revenue from ALL customers who churned yes
print(f"Segment Share of Total Churned Revenue: {(segment['MonthlyCharges'].sum()/tot_churned_yes_rev)*100:.2f}%")
#perc of tot churned revenue that came from our specific segment
each_segment = (df[df["Churn"] == "Yes"]
    .groupby(["TenureGroup", "Contract"], observed=True)["MonthlyCharges"].sum())
#Only churned(Yes) customers' monthly revenue for each segment(tenure+contract)
print("=== Churned Revenue by Segment (Raw) ===")
print(each_segment)
each_segment_revenue = each_segment.reset_index(name="ChurnedRevenue")
#  TenureGroup    Contract          ChurnedRevenue
#0  0-11 Months   Month-to-month     65672.30
#1  0-11 Months      One year         350.00
each_segment_revenue["Segment"]= (each_segment_revenue["TenureGroup"].astype("str")+
        "|"+each_segment_revenue["Contract"])
each_segment_revenue=each_segment_revenue.sort_values("ChurnedRevenue", ascending=False)
#segment=tenure+contract
print("=== Churned Revenue by Segment (Sorted) ===")
print(each_segment_revenue)
new_dataframe=each_segment_revenue[["Segment","ChurnedRevenue"]]
print("=== Segment & Churned Revenue (Final Table) ===")
print(new_dataframe)
plt.barh(each_segment_revenue["Segment"], each_segment_revenue["ChurnedRevenue"])
plt.xlabel("ChurnedRevenue")
plt.ylabel("Segment")
plt.title("Churned Monthly Revenue by Customer Segment")
plt.tight_layout()
plt.savefig("churned_revenue_by_segment.png", dpi=300, bbox_inches = "tight")
plt.show()
#ChurnedRevenue= sum of the MonthlyCharges(of each cus) of customers who churned, within each segment
median = segment["MonthlyCharges"].median()# $X
print(f"Median Monthly Charge: {median:.2f}")
segment_rev_above_median = segment["MonthlyCharges"][segment["MonthlyCharges"]>median].sum() # $Y
#1st select the col then filter
print(f"Churned Monthly Revenue(x) : {segment_rev_above_median:.2f}")
perc_Y = (segment_rev_above_median/tot_churned_yes_rev)*100
print(f"Share of Total Churned Revenue(Y): {perc_Y:.2f}%")
