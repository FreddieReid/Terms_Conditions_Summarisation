

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


get_ipython().ast_node_interactivity = 'all'

#get own data

cols1 = ['document_ref', 'original_text', 'extractive_summary', 'abtractive_summary']

own_data = pd.read_csv('../input/summariesd/terms_summarisations.csv', usecols = cols1)

own_data.isnull().sum()

# create binary importance classifier


own_data["labels"] = np.where(own_data['abtractive_summary']=='not important', 0, 1)

round(own_data["labels"].value_counts(normalize = True),3)*100

cols = ["document_ref", "original_text", "labels"]

data = pd.read_csv('../input/manorandlis/manor_li_data.csv', usecols = cols)

total_data = pd.concat([data, own_data])

from sklearn.model_selection import train_test_split
 
X = total_data["original_text"]
y= total_data["labels"]   
    
#split into train and test set with stratification on y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


#split test set into validation and test set

X_val = X_test[:125]
X_test = X_test[125:]
y_val = y_test[:125]
y_test = y_test[125:]

#concatenate text with the labels

training_data = pd.concat([X_train, y_train], axis = 1)
validation_data = pd.concat([X_val, y_val], axis = 1)
test_data = pd.concat([X_test, y_test], axis = 1)







vectorizer = TfidfVectorizer()

df_train_vectorizer = vectorizer.fit_transform(training_data['original_text'])
df_dev_vectorizer = vectorizer.transform(validation_data['original_text'])
df_test_vectorizer = vectorizer.transform(test_data['original_text'])

log_vec = LogisticRegression(random_state=42, solver = "liblinear")
log_vec.fit(df_train_vectorizer, training_data["labels"])

#create predictive values

pred_train_vec = log_vec.predict(df_train_vectorizer)
pred_dev_vec = log_vec.predict(df_dev_vectorizer)
pred_test_vec = log_vec.predict(df_test_vectorizer)


#find accuracy scores

training_score_vec = accuracy_score(training_data["labels"], pred_train_vec)
print("Training accuracy is {:.3f}".format(training_score_vec))


dev_score_vec = accuracy_score(validation_data["labels"], pred_dev_vec)
print("Validation accuracy is {:.3f}".format(dev_score_vec))
dev_score_vec = precision_score(validation_data["labels"], pred_dev_vec)
print("Validation accuracy is {:.3f}".format(dev_score_vec))
dev_score_vec = recall_score(validation_data["labels"], pred_dev_vec)
print("Validation accuracy is {:.3f}".format(dev_score_vec))
dev_score_vec = f1_score(validation_data["labels"], pred_dev_vec)
print("Validation accuracy is {:.3f}".format(dev_score_vec))

test_score_vec = accuracy_score(test_data["labels"], pred_test_vec)
print("test accuracy is {:.3f}".format(test_score_vec))

test_score_vec = precision_score(test_data["labels"], pred_test_vec)
print("test precision is {:.3f}".format(test_score_vec))

test_score_vec = recall_score(test_data["labels"], pred_test_vec)
print("test recall is {:.3f}".format(test_score_vec))

test_score_vec = f1_score(test_data["labels"], pred_test_vec)
print("test f1 is {:.3f}".format(test_score_vec))


import eli5
eli5.show_weights(estimator=log_vec, feature_names= list(vectorizer.get_feature_names()),top=(10, 10))




from sklearn.svm import SVC

vectorizer = TfidfVectorizer()

df_train_vectorizer = vectorizer.fit_transform(training_data['original_text'])
df_dev_vectorizer = vectorizer.transform(validation_data['original_text'])
df_test_vectorizer = vectorizer.transform(test_data['original_text'])


svc_vec = SVC(random_state=42, kernel = "poly", degree = 2)
svc_vec.fit(df_train_vectorizer, training_data["labels"])

#create predictive values

pred_train_vec = svc_vec.predict(df_train_vectorizer)
pred_dev_vec = svc_vec.predict(df_dev_vectorizer)
pred_test_vec = svc_vec.predict(df_test_vectorizer)


#find accuracy scores

training_score_vec = accuracy_score(training_data["labels"], pred_train_vec)
print("Training accuracy is {:.3f}".format(training_score_vec))


dev_score_vec = accuracy_score(validation_data["labels"], pred_dev_vec)
print("Validation accuracy is {:.3f}".format(dev_score_vec))
dev_score_vec = precision_score(validation_data["labels"], pred_dev_vec)
print("Validation precision is {:.3f}".format(dev_score_vec))
dev_score_vec = recall_score(validation_data["labels"], pred_dev_vec)
print("Validation recall is {:.3f}".format(dev_score_vec))
dev_score_vec = f1_score(validation_data["labels"], pred_dev_vec)
print("Validation f1 is {:.3f}".format(dev_score_vec))

test_score_vec = accuracy_score(test_data["labels"], pred_test_vec)
print("test accuracy is {:.3f}".format(test_score_vec))

test_score_vec = precision_score(test_data["labels"], pred_test_vec)
print("test precision is {:.3f}".format(test_score_vec))

test_score_vec = recall_score(test_data["labels"], pred_test_vec)
print("test recall is {:.3f}".format(test_score_vec))

test_score_vec = f1_score(test_data["labels"], pred_test_vec)
print("test f1 is {:.3f}".format(test_score_vec))
