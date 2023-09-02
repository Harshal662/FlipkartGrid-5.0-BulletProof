# BulletProof - A Fashion recommendation System

## Backend folder file structure
![Alt text](<Backend Folder file structure.jpg>)

## API Reference

#### Get 100 best selling Products- Popularity based recommendation

```
  GET /api/bestsellers
```

#### Get BestSelling products for men- Popularity based recommendation

```
  GET /menProducts
```

#### Get BestSelling products for women- Popularity based recommendation

```
  GET /womenProducts
```

#### Get all products data

```
  GET /allProducts
```

#### Get Similar Items - Content Based Recommandation

```
  GET /prod/${title}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `title`      | `string` | Title of the product from which fetch all similar products


#### Get Recommandation of products based on ratings- Collaborative Filtering

```
  GET /recommand/${title}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `title`      | `string` | Title of the product from which fetch related products which user should buy based on ratings


### get the similar products available on website based on image- Fashion recommandation using Resnet model

```
  GET /imageData
```

### Send uploaded image data to backend for image preprocessing, feature extraction and recommandation

```
  POST /imageData
```
