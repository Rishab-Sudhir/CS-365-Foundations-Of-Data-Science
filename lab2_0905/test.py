import requests

url = "https://newsapi.org/v2/everything?q=tesla&from=2023-08-29&sortBy=publishedAt&apiKey=bb0637e927f8495db2366909e4115e6f"

response = requests.get(url)

if response.status_code == 200:

    data = response.json()
    print(data) 


