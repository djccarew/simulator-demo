# Simulator Demo

Add the pip requirements and run the following 

```
gunicorn -b localhost:5000 --workers 1  --threads 3 wscommentary:app
```

Use a web socket client and send messages to 

http://127.0.0.0:5000/watsonx

For example:
```
{ "type": "ping" }
```
