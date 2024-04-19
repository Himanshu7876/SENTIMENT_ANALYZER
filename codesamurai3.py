import pandas as pd
import streamlit as st

st.set_page_config(page_icon=":)", page_title="Sentiment Analyzer", layout="centered")

@st.cache(allow_output_mutation=True)
def train_model(X, Y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    from sklearn.model_selection import GridSearchCV
    model = MultinomialNB()
    param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_tfidf, Y)
    best_params = grid_search.best_params_
    best_nb_classifier = MultinomialNB(alpha=best_params['alpha'])
    best_nb_classifier.fit(X_tfidf,Y)
    return best_nb_classifier, vectorizer

@st.cache
def predict_sentiment(text, model, vectorizer):
    tweet_vectorized = vectorizer.transform([text])
    prediction = model.predict(tweet_vectorized)
    return prediction[0]

def save_feedback(feedback):
    try:
        df = pd.read_csv("feedback2.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["feedback"])

    new_entry = pd.DataFrame({"feedback": [feedback]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv("feedback2.csv", index=False)
    st.success("Feedback saved successfully")

data = pd.read_csv('final_data2.csv')
data = data.drop(data.columns[0], axis=1)
data.rename(columns={'0': 'sentiments'}, inplace=True)
data.dropna(inplace=True, axis=0)

X = data['Tweets']
Y = data['sentiments']

model, vectorizer = train_model(X, Y)

st.title("Sentiment Analyzer")
st.markdown("---")

text = st.text_area("Enter text here")
submit = st.button("Analyze")

if submit:
    if text.strip() == "":
        st.error("Please enter some text")
    else:
        output = predict_sentiment(text, model, vectorizer)
        combine="Predicted sentiment : "+output
        st.success(combine)

        
feedback = st.text_area("Enter feedback")
button_1 = st.button("Submit Feedback")
if button_1:
    if feedback.strip() == "":
        st.error("Please enter some feedback")
    else:
            save_feedback(feedback)
            feedback = st.text_area("Enter feedback", value="")
