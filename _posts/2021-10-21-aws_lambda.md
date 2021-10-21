---
title: 'A Serverless _"Hello, From Lambda!"_ with AWS Lambda'
date: 2021-10-21
permalink: /posts/2021/10/aws_lambda/
tags:
  - aws
  - lambda
  - aws lambda
  - serverless
  - cloud watch
---

#  **A Serverless _"Hello, From Lambda!"_ with AWS Lambda**


<p align="center">
<img src="https://thumbs.gfycat.com/FlawlessFlusteredCony-size_restricted.gif" alt="Source: https://www.youtube.com/watch?v=eOBq__h4OJ4">
</p>
<p align="center">
<caption> Source: https://www.youtube.com/watch?v=eOBq__h4OJ4 </caption>
</p>





AWS Lambda lets you run code without you thinking about servers. This means Lambda is a compute service that enables you to run code without you managing the servers. You can learn more about lambda on [AWS Lambda](https://aws.amazon.com/lambda/)


# **Why we don't want to manage the servers?**



Application needs backend code to respond to events on the UI. The events could be file upload, image upload, in-app activity like button clicks etc.

 To host and run backend code on every event requires scalable servers, operating systems, databases etc. We also need one or more server and load balancers to provide service to each request without delay. Additionally, we need to apply security mechanisms and patches and monitor everything to ensure availability and high performance.




<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1OiFH711JZMl9z5YlBk_XQrc4qdSCRzBn" alt="Server management"/>
</p>


Using AWS Lambda, developers can focus on application development without managing a lot of time on the server.

# **Implementation**

In this section, we will learn the basics of running code on AWS Lambda without provisioning or managing servers. __Let's implement a "Hello from the Lambda!"  using the AWS Lambda.__

Requirements:
* AWS login credentials



## **Enter the AWS Management Console**

Open the AWS Management Console in a browser and then search Lambda on the search bar as shown in the following figure:


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1VkjVLWt41XVmsZ4qi1IiKB9cjHWSIkri"/>
</p>


When you search the Lambda, the following tiles appear. Click the Lambda as shown in the following figure.



<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1BUYwriUR-xXhEFQV6gNfWy92u8NYQuqH"/>
</p>

After clicking the Lambda, the following dashboard appears.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1vAQrU4xykDFjjJ01q4fqvJTflNQFS6zu"/>
</p>

The dashboard shows the count of active lambda functions, the storage size taken by the Lambda etc. Currently, I have no lambda function deployed; therefore, the value is zero.  In this tutorial, we are only interested in the lambda function so let's jump into it by clicking the function on the left side.

## **Create the Lambda Function**

Click on create function to create the lambda function. 


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1A7ASapJoS_LfnKyNvi6zU0W9pIqFHur3"/>
</p>



 After clicking the button, the following pop-up appears.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1Zp-tGT4c1TtNOOAGLfRsem-cSUtEVbEQ"/>
</p>

In this tutorial, we will author a lambda function from scratch. Select the option and scroll down. However, we will discuss other options in some other tutorials.




<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1nsKjeXyfcYqHhdPGW8p16dvw1mi6D9C7"/>
</p>

A lambda function requires a unique name, runtime, instruction set architecture etc. Let's fill in the necessary details. I will name my lambda function as *blog_test_lambda* and choose python 3.8 as runtime and x86_64 bit architecture as shown in the above figure. You can provide your own name, architecture and runtime.  Note that other fields are left as default. The important part for you to explore is the execution role.

*A Lambda function's execution role is an AWS Identity and Access Management (IAM) role that grants the function permission to access AWS services and resources.*

Click on create function.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=12XeNUAZnhs_4_7NH-9Pi0uXR7ect-HwF"/>
</p>

The function is successfully created. 


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1pJ_Iuruw9yxetDhHJDsVM1c5mAbL1fLM"/>
</p>

## **Configure the Lambda Function**

Please scroll down to the code section to see the default code provided to us. 

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1BMWNl9GbhlhAkl3TWJjBdz3Fl16z452d"/>
</p>

We will get back to the code section later but first, let's explore the configuration tab. 

Click on the configuration tab to configure the lambda function. On the general configuration tab, the default memory allocated to the lambda function is 128MB, and the timeout is 3 seconds. Let's try to change it by clicking on the edit button. 

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1qu_Hm5p6Nvp5vKeCgg7LQKhvP_EoPVHl"/>
</p>

Let's change the default memory and timeout value and set it to 200 MB and 2 minutes, respectively. After doing that, click on the Save button.


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1d8kZSA-975SkfJQMD_baHnudCGtTmCNi"/>
</p>

There are other tabs like triggers destinations. We didn't define any triggers and destinations; therefore, the section is blank, as shown in the following diagram. 


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1t230JUbkaZD0BNSH2eeqV7BxgWhJRyHk"/>
</p>


Another essential configuration setting is Environment variables. 

We can use environment variables to adjust our function's behaviour without updating code. An environment variable is a key-value pair that is stored in a function's version-specific configuration.

* We can create an environment variable for our function by defining a key and a value. Your function uses the name of the key to retrieve the value of the environment variable. 

Click on edit to add an environment variable.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1I13L8Tnqd2Tph_PpoFZu6LOXSAbxRrVK"/>
</p>


After clicking on edit, the following pop-up appears. Click on the "Add environmet variable" button.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1ITsQbUG-55YHakXnj4yKrc0Y30OOLclg"/>
</p>


Provide the key-value pair and then click on Save. We will try to access and print the value of a key "Kathmandu" on the code section later. 


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1ybpWbhXUskxFAJ7WMpkSNenymV1sULh-"/>
</p>

The last configuration I would like to discuss is the asynchronous invocation. 

* *When we invoke a function asynchronously, we don't wait for a response from the function code. We hand off the event to Lambda, and Lambda handles the rest. However, we can configure how Lambda handles errors and can send invocation records to a downstream resource to chain together components of our application. For asynchronous invocation, Lambda places the event in a queue and returns a successful response without additional information.  (Src: AWS lambda Asynchronous invocation)*

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1eHt_CBluvxuzljjzzEpwmocmalT5O25g"/>
</p>

we can Configure the following settings:

* Maximum age of event – The maximum amount of time Lambda retains an event in the asynchronous event queue, up to 6 hours.

* Retry attempts – The number of times Lambda retries when the function returns an error, between 0 and 2.

When an async invocation event exceeds the maximum age or fails all retry attempts, Lambda discards it. To retain a copy of discarded events, we can configure a failed-event destination, or we can configure the function with a dead-letter queue to save discarded events for further processing.


## **Invoke the Lambda Function**

Now let's go back into the code tab and write some code. 

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1BMWNl9GbhlhAkl3TWJjBdz3Fl16z452d"/>
</p>

In the above function, we will try to print three text
1. lambda function is invoked. 
2. Value of environment variable for a key "Nepal" that we just configured.

3. Value in events or the input given in lambda function. 



When you make the changes in the code, you will see a warning that says "changes not deployed". Therefore you should save or deploy changes after you are ready to test the lambda function. To do so, click on deploy, as shown in the following figure.


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1TspZFEd-fzhILXMTgvZ_WbxNgPW_1jDl"/>
</p>

`Note that the lambda function returns the dictionary written on the return statement. `

The changes are now deployed. Click on test to test the lambda function.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1l_bmLX-_Wz3k42mBM2dXFlFrwY0YkLHk"/>
</p>


After clicking on Test, the configuration of the test event appears. In this, you need to provide the Event name only. I have given the name blog_test_lambda_input. After providing the name, scroll down and click on create button. 

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1TwmgawpiTal6I8gO05FtGZfwZ1qpAXp4"/>
</p>


The click on the Test button results in the following output. 

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1JC4mPprC0Y1A8-dZz3nDoe1CxlmIjl6r"/>
</p>


The execution results show the Test event name, response or the return statement of the lambda function along with function logs and requestId.

## **Monitor the Lambda Function**

The print statement we wrote in on Function logs. Remember, we printed the event on the data received to the Lambda. The event is a JSON that we sent while configuring the test event. The details of the logs can also be visualized on amazon cloud watch.


* *AWS Lambda automatically monitors Lambda functions on your behalf, reporting metrics through Amazon CloudWatch. To help you troubleshoot failures in a function after you set up permissions, Lambda logs all requests handled by your function and also automatically stores logs generated by your code through Amazon CloudWatch Logs. (aws.amazon.com)*

**Let's visualize the amazon cloud watch for our lambda function.**

Click the monitor and then click on view logs in CloudWatch.


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1-Xij1APplSZs6O-fXZYLbp8TIeMLkQGL"/>
</p>

A new tab will open where you will see a page that looks like the following image.


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1WrAiVamGcAZrueototxGGBfiKM6t6lGa"/>
</p>

There is one file in a log stream. Let's click the file to open it.  

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1dWPP8U_4YUEsU_rCWZV3B2BBpsg6ixyI"/>
</p>

Now you can see a detailed log report on the lambda function. 

# **Conclusion**



In this entire tutorial, we have learned about Lambda and why we want to use Lambda. Additionally, we learned how to create, invoke and monitor a serverless hello world with AWS Lambda.

This is a learning series. Therefore I would like to show you how to delete Lambda as a bonus topic. 

# **Bonus !!!**

## **How to delete the Lambda Function**

1. Close the cloud Watch tab and go to the code section and then scroll up. 

2. Click on Lambda as shown in the figure.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1gWd_JdjsbIr9dahTGS97KnOOQwcVnivK"/>
</p>


3. After clicking the Lambda, you will see a list of lambdas you have created. Tick mark the Lambda you want to delete and click on the action to delete the Lambda, as shown in the following figure.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1Y-WbdGeiv51Tnol4HXEe-FxKIQ_4glKC"/>
</p>


# References

[1] https://aws.amazon.com/lambda/
[2] https://aws.amazon.com/getting-started/hands-on/run-serverless-code/
