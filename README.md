
# BulletProof - A Fashion recommendation System

The recommendation algorithms used in the development of BulletProof include collaborative filtering, content-based filtering, and popularity-based filtering.

The Popularity-Based Recommender System operates on the tenets of trendiness and/or popularity. It generates popularity among users by making product recommendations based on the total number of product ratings and the average user rating.

Content-Based The comparable content theory underlies how recommender systems operate. It makes product recommendations based on comparable product content. Based on cosine similarity, the model suggests related products to the user.


## Goals of the project

  This project's major objective is to create a personalized fashion suggestion system that may assist users in finding clothing items that suit their tastes. The system also seeks to offer a user interface that is engaging and simple to use.

## Process

  The dataset of 17000 Myntra items was assembled and prepared as the initial stage in building this project. Next, this dataset was used to train a variety of machine learning methods, including hybrid, collaborative filtering, popularity-based, and ResNet-based feature extraction. These features and algorithms were subsequently added to the Flask backend.

  An interactive user interface was made for the front end using Reactjs and Bootstrap. To provide safe user authentication, Firebase authentication was also implemented into the system.

## Features

  1. Image-based fashion recommendation: 
    Users can input pictures of fashion items they like, and the system will suggest related items based on feature extraction performed using ResNet.

  2. Recommendation algorithms: 
    To give customers a range of recommendations, the website employs many recommendation algorithms, including hybrid, collaborative filtering, and popularity-based algorithms.

  3. Personalised suggestions: 
    Based on users' preferences and previous experiences with the website, the website makes personalized recommendations to users.

  4. Personalised ChatBot: 
    The service offers a genuine, individually trained GenerativeAI ChatBot that will provide outfit recommendations depending on various conditions.

  5. Secure authentication: 
    To guarantee that only authorized users may access the website, Firebase authentication is employed.

  6. Interactive user interface: 
    Reactjs and Bootstrap were used to construct the website's interactive user interface, which makes it simple for people to browse and utilize.





## Resources

The Dataset used in this project is taken from Kaggle. It is a product listing from Myntra.com for the period of June 2019 to August 2019.
This dataset holds 15K records.

Dataset Link: https://www.kaggle.com/datasets/promptcloud/all-products-from-myntracom-2019

## Tech Stack

**Client:** React, Material-UI, react-bootstrap

**Server:** Python, Flask

**Database:** Firebase

## Run Locally

Clone the project

```bash
  git clone https://github.com/Harshal662/FlipkartGrid-5.0-BulletProof.git
```

Go to the project directory

```bash
  cd FlipkartGrid-5.0-BulletProof
```

Go to the requirements.txt file in the same folder and install all required dependencies.

### Note - 
    There are a total of 3 requirement.txt (root folder, backend folder, ChatBot folder) files install each of them for the proper functioning of the project.

```bash
  npm install
```

## Environment Variables for Firebase

To run this project, you will need to add the following environment variables to your firebase.js file

`API_KEY`: "",

`authDomain`: "",

`projectId`: "",

`storageBucket`: "",

`messagingSenderId`: "",

`appId`: "",

`measurementId`: "",


Start the server

```bash
  npm run dev 
```

Runs the app in the development mode.
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

## For Backend and ChatBot 
### If there are any conflicts in installing dependencies create a virtual environment for them.

Open a new terminal and run

```bash
  cd backend
```

Create a Virtual Environment in Python

```bash
  pip install virtualenv
```
```bash
  virtualenv env
```

For Windows
```bash
  .\env\Scripts\activate
```

For Mac

```bash
    source env/bin/activate
```

Install all Python modules which are in the backend/requirements.txt file

```bash
    pip install -r requirements.txt
```

### Repeat the same process for ChatBot

#### NOTE

Download backend files from here: 

Add these files inside the backend folder. As models were large they can't be uploaded on Git Hub. Please download it from the Google Drive Link.



## Feature Engineering
Due to the abundance of null values in my Myntra dataset and the inadequacy of the column data for model training, feature engineering of the dataset was required for effective model training. To preprocess the data, I have developed several functions for a given column. 

## Recommendation Based on Popularity:
The popularity-based recommendation system, as its name implies, follows trends. I discovered the total number of ratings on each product as well as the average ratings of items by using the rating dataset. After that, I divide the items into groups based on thresholds that I applied to the overall ratings, and I arranged the products according to decreasing average ratings. I acquired my top 100 goods in this way.

## Content-Based Recommendation: 
This recommender system uses the product's content (actual_color, dominant_color, product_type, product_details, complete_the_look, inventory, specs, etc.) to determine if it is comparable to other items.

By preprocessing the text, performing lemmatization, and eliminating stopwords from a corpus of data using the nltk (natural language processing toolkit) package, I consolidated the content of the product into a single corpus.

I next turn my corpus of data into vectors using TfidfVectorizer in the Sklearn module.

Now that I've used sigmoid_kernel with sklearn, I can compare goods based on their cosine similarity.

## Similarity score
How does it determine which item is most comparable to the preferred item by the user? The similarity scores are now shown.

It is a numerical number that spans from 0 to 1, used to assess how similar two objects are to one another on a scale from 0 to 1. By comparing the text details of the two items' details, this similarity score is calculated. Therefore, the similarity score is a metric used to compare two things supplied textual descriptions. Cosine-similarity can be used to achieve this.

Then, the merchandise that is most likely to be comparable is suggested.

## How does Cosine Similarity Work??

![Cosine Similarity](https://i0.wp.com/clay-atlas.com/wp-content/uploads/2020/03/cosine-similarity-2.png?fit=800,208&ssl=1) 

Regardless of the size of the documents, the cosine similarity measure is employed to determine how similar they are. It makes a mathematical estimation of the cosine of the angle created by two vectors projected in a multidimensional space. Even if two comparable documents are spaced apart by the Euclidean distance (because of the size of the documents), they are likely to be oriented closer together because of the cosine similarity. The angle is narrower the higher the cosine similarity.

![Cosine Similarity](https://i0.wp.com/towardsmachinelearning.org/wp-content/uploads/2020/07/image-5.png?w=1170&ssl=1)

## Collaborative Filtering
The only additional need for collaborative filtering is users' past preferences for a particular group of items. The fundamental premise here is that users who have agreed in the past are likely to agree in the future since it is based on previous data.

The goal is to identify the users who are closest to your target user (also known as closest neighbors) and use their ratings of an item as a predictor of the target user's rating of that item.

## Matrix Factorization
A user's alignment with a set of latent characteristics and the degree to which a movie fits into this collection of latent features are the final results of matrix factorization. 
 
![Cosine Similarity](https://datasciencedojo.com/wp-content/uploads/content-based-recommendation-system-suggestions.webp)

## Product Recommendation from Image
A trained image processing model is called Resnet. To extract features from the photographs in my dataset, I utilized resnet, the foundation of computer vision tasks. Using the resnet model, I also extract characteristics from submitted photos. I then obtain the closest comparable goods using the Nearest Neighbour classifier and display them to the user.





