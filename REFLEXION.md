question 1 :How do we officially designate the model
to use?
answer1:With the champion alias: models:/churn-model@champion points to
the chosen version, and it can be moved in one line

question 2: How can an application use the model?
answer2: With a web request to the API: POST /predict with a JSON, the response

contains the prediction

question 3 What happens when we change
champion?
answer3:We move the alias and restart the API: the code of the API and of the
client applications does not change
