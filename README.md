# credit-application-service
credit-application-service is a service with a simple REST API that provides recommendation / prediction for single credit application request based on machine learning.

The project aims to predict how likely a credit card request will get approved based on age, gender, credit score, income, debt, etc.

Banking industries receive so many applications for credit card request. Going through each request manually can be very time consuming, also prone to human errors. However, if we can use the historical data to build a model which can shortlist the candidates for approval that can be great. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays.

## DataSet Source
The dataset is Credit Card Approval dataset from the UCI Machine Learning Repository. The data consists of 16 columns and 690 records.
This file concerns credit card applications.  All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

https://archive.ics.uci.edu/dataset/27/credit+approval

## Data usage
2% of the data is omitted and some of it is used bellow to test the API

``` 
287  a,?,1.5,u,g,ff,ff,0,f,t,02,t,g,00200,105,-
656  a,21.08,5,y,p,ff,ff,0,f,f,0,f,g,00000,0,-
320  b,36.75,0.125,y,p,c,v,1.5,f,f,0,t,g,00232,113,+
212  b,24.33,6.625,y,p,d,v,5.5,t,f,0,t,s,00100,0,+
```

## Resources Used

``` 
Flask==3.0.0
marshmallow==3.20.2
numpy==1.26.3
pandas==2.1.4
scikit_learn==1.3.2
matplotlib~=3.8.3
pytest~=8.1.0
```

## Running the app

### Using docker
Build the docker image
```
docker build -t credit-application-service-img .
```
Run docker 
```
docker run -p9000:5000 --name cas credit-application-service-img
```

### Run locally
``` 
python3 -m flask --app app run
```

## Using the API

### Getting info regarding score and confusion matrix

Request:
``` 
curl -i 'localhost:9000/info'
```

Response:

``` 
{"matrix":"[[126   0]\n [  0 101]]","score":1.0}
```

### Getting a recommendation for new credit application

#### Credit Application Test 1
``` 
287  a,?,1.5,u,g,ff,ff,0,f,t,02,t,g,00200,105,-
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"a","p1":1,"p2":1.5,"p3":"u","p4":"g","p5":"ff","p6":"ff","p7":0,"p8":"f","p9":"t","p10":"02","p11":"t","p12":"g","p13":"00200","p14":105}'
```
Response:
``` 
{"status":"DECLINED"}
```

#### Credit Application Test 2
``` 
656  a,21.08,5,y,p,ff,ff,0,f,f,0,f,g,00000,0,-
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"a","p1":21.08,"p2":5.0,"p3":"y","p4":"p","p5":"ff","p6":"ff","p7":0.000,"p8":"f","p9":"f","p10":"0","p11":"f","p12":"g","p13":"00000","p14":0}'
```
Response:
``` 
{"status":"APPROVED"}
```

#### Credit Application Test 3
``` 
257  b      20.00  0.0  u  g   d   v  0.5  f  f   0  f  g  00144    0  -
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"b","p1":20.00,"p2":0.0,"p3":"u","p4":"g","p5":"d","p6":"v","p7":0.5,"p8":"f","p9":"f","p10":"0","p11":"f","p12":"g","p13":"00144","p14":0}'
```
Response:
``` 
{"status":"DECLINED"}
```

#### Credit Application Test 3
``` 
165  a      40.83  10.000  u  g   q   h  1.750  t  f   0  f  g  00029  837  +
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"a","p1":40.83,"p2":10.000,"p3":"u","p4":"g","p5":"q","p6":"h","p7":1.750,"p8":"t","p9":"f","p10":"0","p11":"f","p12":"g","p13":"00029","p14":837}'
```
Response:
``` 
{"status":"APPROVED"}
```