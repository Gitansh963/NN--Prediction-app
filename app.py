from flask import Flask, request, render_template,send_file
import mammoth
import joblib
import numpy as np
import pandas as pd
import random
app = Flask(__name__)

from keras.models import load_model

model = load_model('best_model_nn.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graphs')
def view_graphs():
    return render_template('graphs.html')

@app.route('/feedback')
def view_feedback():
    return render_template('feedback.html')

@app.route('/report')
def view_report():
    with open('static/Analysis_Report.docx', 'rb') as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html = result.value
    return render_template('report.html', html=html)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        first_term_gpa = float(request.form['first_term_gpa'])
        second_term_gpa = float(request.form['second_term_gpa'])
        first_language = request.form['first_language']
        funding = request.form['funding']
        fastTrack = request.form['fastTrack']
        coop = request.form['coop']
        residency = request.form['residency']
        gender = request.form['gender']
        previous_education = request.form['previous_education']
        age_group = request.form['age_group']
        high_school_average_marks = int(request.form['high_school_average_marks'])
        math_score = int(request.form['math_score'])
        english_grade = request.form['english_grade']
        data = {
            'First Term Gpa': normalized_value(first_term_gpa,4.5),
            'Second Term Gpa': normalized_value(second_term_gpa,4.5),
            first_language: 1,
            funding: 1,
            fastTrack: 1,
            coop: 1,
            residency: 1,
            gender: 1,
            previous_education: 1,
            age_group: 1,
            'High School Average Mark': normalized_value(high_school_average_marks,108),
            'Math Score': normalized_value(math_score,50),
            english_grade: 1
        }
        data_cleaned = clean(data)
        df = pd.DataFrame([data_cleaned])
        print(df.shape)
        print(df)
        prediction = result1(df)
        prediction2 = str(prediction)
        print(prediction2)
        final_result = ' '.join(map(lambda value: '1' if float(value) > 0.1 else '0', prediction2[2:-2].split()))
        print(final_result)
        string_result = ''
        if(final_result == '0 1'):
            string_result = 'You are likely to pass the course.'
        elif(final_result == '1 0'):
            string_result = 'You are likely to drop the course.'
        print(string_result)
        return render_template('result.html',  input = data_cleaned, id = df, result_prediction = string_result)
    

# 0,1 = Passed the course 
# 1,0 = Dropped the course
            
def clean(data):
    data_dictionary = {
        'First Term Gpa': 0,
        'Second Term Gpa': 0,
        'High School Average Mark': 0,
        'Math Score': 0,
        'First Language_1': 0,
        'First Language_2': 0,
        'First Language_3': 0,
        'Funding_1': 0,
        'Funding_2': 0,
        'Funding_3': 0,
        'Funding_4': 0,
        'Funding_5': 0,
        'Funding_6': 0,
        'Gender_1': 0,
        'Gender_2': 0,
        'Gender_3': 0,
        'Previous Education_0': 0,
        'Previous Education_1': 0,
        'Previous Education_2': 0,
        'Age Group_1': 0,
        'Age Group_2': 0,
        'Age Group_3': 0,
        'Age Group_4': 0,
        'Age Group_5': 0,
        'Age Group_6': 0,
        'Age Group_7': 0,
        'Age Group_8': 0,
        'English Grade_1': 0,
        'English Grade_2': 0,
        'English Grade_3': 0,
        'English Grade_4': 0,
        'English Grade_5': 0,
        'English Grade_6': 0,
        'English Grade_7': 0,
        'English Grade_8': 0,
        'FastTrack_1': 0,
        'FastTrack_2': 0,
        'Coop_1': 0,
        'Coop_2': 0,
        'Residency_1':0,
        'Residency_2':0
    }
    dict3 = {k: float(data[k]) if k in data else float(data_dictionary[k]) for k in data_dictionary}

    return dict3

def result1(data):
    prediction = model.predict(data)
    return prediction

def normalized_value(value, max_value):
    if value > max_value:
        diff = value - random.uniform(0, max_value)
        normalized_value = (diff/max_value) if max_value != 0 else 0
    else:
        normalized_value = (value/max_value) if max_value != 0 else 0

    normalized_value = max(0, min(1, normalized_value))

    return normalized_value

if __name__ == '__main__':
    app.run(debug=True, port = 5000)