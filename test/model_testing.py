import pandas as pd
import joblib

# load the saved model
model = joblib.load('model/trained_model.pkl')

print('Model loaded successfully!')
print('Type - exit - to exit prediction testing!')

while True:
    title = input(f'Enter review title:')
    if title.lower() == 'exit':
        print(f"Disconnecting...")
        break
    
    text = input(f'Enter review text:')
    if text.lower() == 'exit':
        print(f"Disconnecting...")
        break

    # compute review length
    review_length = len(text)

    # create a dataframe from input
    user_input = pd.DataFrame([{
        'review_title': title,
        'review_text': text,
        'review_length': review_length
    }])

    # predict sentiment
    prediction = model.predict(user_input)[0]
    print(f"Predict sentiment: {prediction}\n" + "-" * 30)