# credit-application-service
credit-application-service is a service with a simple REST API that provides recommendation / prediction for single credit application request based on machine learning.

The project aims to predict how likely a credit card request will get approved based on age, gender, credit score, income, debt, etc.

Banking industries receive so many applications for credit card request. Going through each request manually can be very time consuming, also prone to human errors. However, if we can use the historical data to build a model which can shortlist the candidates for approval that can be great. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays.

## DataSet Source
The dataset is Credit Card Approval dataset from the UCI Machine Learning Repository. The data consists of 16 columns and 690 records.
This file concerns credit card applications.  All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

https://archive.ics.uci.edu/dataset/27/credit+approval

## Data usage
1% of the data is used as test data and used bellow to test the API

``` 
287 b,29.50,0.58,u,g,w,v,0.29,f,t,01,f,g,00340,2803,-
513 b,20.25,9.96,u,g,e,dd,0,t,f,0,f,g,00000,0,+
258 a,20.75,9.54,u,g,i,v,0.04,f,f,0,f,g,00200,1000,-
338 a,33.25,3,y,p,aa,v,2,f,f,0,f,g,00180,0,-
320 b,21.25,1.5,u,g,w,v,1.5,f,f,0,f,g,00150,8,+
212 b,60.08,14.5,u,g,ff,ff,18,t,t,15,t,g,00000,1000,+
626 b,23.75,12,u,g,c,v,2.085,f,f,0,f,s,00080,0,-
```

y_test (test targets)
```
287    0
513    1
258    0
338    0
320    1
212    1
626    0
```

y_pred (predictions on test data)
``` 
[0 1 0 0 0 1 0]
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
{"matrix":"[[106  28]\\n [  8  99]]","score":0.8506224066390041}
```

### Getting a recommendation for new credit application

#### Credit Application Test 1
``` 
287 b,29.50,0.58,u,g,w,v,0.29,f,t,01,f,g,00340,2803,-
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"b","p1":29.50,"p2":0.58,"p3":"u","p4":"g","p5":"w","p6":"v","p7":0.29,"p8":"f","p9":"t","p10":"01","p11":"f","p12":"g","p13":"00340","p14":2803}'
```
Response:
``` 
{"status":"DECLINED"}
```

#### Credit Application Test 2
``` 
513 b,20.25,9.96,u,g,e,dd,0,t,f,0,f,g,00000,0,+
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"b","p1":20.25,"p2":9.96,"p3":"u","p4":"g","p5":"e","p6":"dd","p7":0,"p8":"t","p9":"f","p10":"0","p11":"f","p12":"g","p13":"00000","p14":0}'
```
Response:
``` 
{"status":"APPROVED"}
```

#### Credit Application Test 3
``` 
258 a,20.75,9.54,u,g,i,v,0.04,f,f,0,f,g,00200,1000,-
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"a","p1":20.75,"p2":9.54,"p3":"u","p4":"g","p5":"i","p6":"v","p7":0.04,"p8":"f","p9":"f","p10":"0","p11":"f","p12":"g","p13":"00200","p14":1000}'
```
Response:
``` 
{"status":"DECLINED"}
```

#### Credit Application Test 4
``` 
338 a,33.25,3,y,p,aa,v,2,f,f,0,f,g,00180,0,-
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"a","p1":33.25,"p2":3,"p3":"y","p4":"p","p5":"aa","p6":"v","p7":2,"p8":"f","p9":"f","p10":"0","p11":"f","p12":"g","p13":"00180","p14":0}'
```
Response:
``` 
{"status":"DECLINED"}
```

#### Credit Application Test 5
False Negative result
``` 
320 b,21.25,1.5,u,g,w,v,1.5,f,f,0,f,g,00150,8,+
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"b","p1":21.25,"p2":1.5,"p3":"u","p4":"g","p5":"w","p6":"v","p7":1.5,"p8":"f","p9":"f","p10":"0","p11":"f","p12":"g","p13":"00150","p14":8}'
```
Response:
``` 
{"status":"DECLINED"}
```


#### Credit Application Test 6
``` 
212 b,60.08,14.5,u,g,ff,ff,18,t,t,15,t,g,00000,1000,+
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"b","p1":60.08,"p2":14.5,"p3":"u","p4":"g","p5":"ff","p6":"ff","p7":18,"p8":"t","p9":"t","p10":"15","p11":"t","p12":"g","p13":"00000","p14":1000}'
```
Response:
``` 
{"status":"APPROVED"}
```

#### Credit Application Test 7
``` 
626 b,23.75,12,u,g,c,v,2.085,f,f,0,f,s,00080,0,-
```
Request:
``` 
curl -X POST 'localhost:9000/creditApplicationRequest' \
--header 'Content-Type: application/json' \
--data '{"p0":"b","p1":23.75,"p2":12,"p3":"u","p4":"g","p5":"c","p6":"v","p7":2.085,"p8":"f","p9":"f","p10":"0","p11":"f","p12":"s","p13":"00080","p14":0}'
```
Response:
``` 
{"status":"DECLINED"}
```