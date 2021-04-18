import requests

url = 'https://api.hgbrasil.com/finance?key=d58315ae'
if 200 <= response.status_code <= 299:
    # Success
    print('Status Code', response.status_code)
    print('Json', response.json())
else:
    # Errors
    print('Status Code', response.status_code)


response = requests.get(url=url)