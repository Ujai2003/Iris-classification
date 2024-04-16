from flask import Flask, render_template, request
import iris_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepallength'])
            sepal_width = float(request.form['sepalwidth'])
            petal_length = float(request.form['petallength'])
            petal_width = float(request.form['petalwidth'])
            
            y_pred = [[sepal_length, sepal_width, petal_length, petal_width]]
            
            trained_model = iris_model.training_model()
            prediction_value = trained_model.predict(y_pred)[0]
            
            setosa = 'The flower is classified as Setosa'
            versicolor = 'The flower is classified as Versicolor'
            virginica = 'The flower is classified as Virginica'
            
            evaluation_metrics = {}
            
            # Evaluate on validation set
            accuracy, precision, recall, f1 = iris_model.evaluate_model(trained_model, iris_model.X_val, iris_model.y_val)
            
            evaluation_metrics['Accuracy'] = accuracy
            evaluation_metrics['Precision'] = precision
            evaluation_metrics['Recall'] = recall
            evaluation_metrics['F1 Score'] = f1
            
            if prediction_value == 0:
                return render_template('index.html', setosa=setosa, metrics=evaluation_metrics)
            elif prediction_value == 1:
                return render_template('index.html', versicolor=versicolor, metrics=evaluation_metrics)
            else:
                return render_template('index.html', virginica=virginica, metrics=evaluation_metrics)
        
        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(error_message)
            return render_template('index.html', error=error_message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
