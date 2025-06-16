import requests

if __name__ == '__main__':
    url = 'https://receptive-jellyfish-production.up.railway.app/api/v1/laisa'
    url_health = 'https://receptive-jellyfish-production.up.railway.app/api/v1/health'

    question = "how can you help me"
    context = ""
    chat_history = []  # Ensure this is a list

    payload = {
        "question": question,
        "chat_history": chat_history,  # This is already a list, so no need for conditional logic
        "context": context
    }

    headers = {'Content-Type': 'application/json'}

    # Print payload for debugging
    print("Payload being sent:", payload)

    # Make the POST request
    response = requests.post(url, json=payload, headers=headers)


    # Print response for debugging
    print("Response status code:", response.status_code)
    print("Response body:", response.text)  # This will show the detailed validation error, if any

    if response.status_code == 200:
        print("Question sent successfully")
        print(response.json())  # Print the JSON response from the API


    else:
        print(f"Failed to send question, status code: {response.status_code}")
