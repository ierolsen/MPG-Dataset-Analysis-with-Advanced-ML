In this project I worked on [Auto-MPG Dataset](https://www.kaggle.com/uciml/autompg-dataset) with Advanced Machine Learning techniques.

During the project, I applied different ML techniques likes Ridge Regression, Losso Regression, Outlier Detection, etc.
I am going to explain something about code at below

----

**Why did I define something called threshold?**

The purpose is more visibility on pair plot and other purpose is minimize the correlation matrix.

![Picture1](https://user-images.githubusercontent.com/30235603/78386294-fe24b300-75e5-11ea-9132-340209245745.png)

By the way if there are more correlated features each other, we can say they are multicollinearity. This situation is disadvantage because we use more features instead of one feature.

----

When the pair plot is drawn, it looks like there are Outliers:

![Picture2](https://user-images.githubusercontent.com/30235603/78386344-11378300-75e6-11ea-8f68-eeff9e936eff.png)

-----
![Picture3](https://user-images.githubusercontent.com/30235603/78386349-172d6400-75e6-11ea-8629-7eeadb7946b2.png)

After Box Plot, I am really sure now, Horsepower and Acceleration have outliers.

**And then I find the outliers:**

![Picture4](https://user-images.githubusercontent.com/30235603/78386362-1dbbdb80-75e6-11ea-96c4-43f2b8730009.png)

![Picture5](https://user-images.githubusercontent.com/30235603/78386377-23b1bc80-75e6-11ea-86dd-562b3c738d67.png)

After finding Outliers, as you can see, shape is reduced. I will apply these codes for acceleration. After all that, last shape is **(392,8)**
It means totally 6 Outliers has been removed with these filters.

------

**Feature Engineering**

**1-)** **Skewness**

![Picture6](https://user-images.githubusercontent.com/30235603/78386392-29a79d80-75e6-11ea-9f95-7d5d26dc4664.png)


3rd momentum of data gives value of Skewness. I will use scipy library for calculating that. (Skewness > 1 = positive, Skewness < -1 = negative)

Also Skewness is named Outliers of our data in tail field.

Using Log Transform we make a transformation from high slope dispersion to low slope dispersion, that mean we can minimize value of skew.

Before Log Transform you can see the Skewness.

![Picture7](https://user-images.githubusercontent.com/30235603/78386403-30361500-75e6-11ea-9f31-2f6c8deecba9.png)

Black one is Normal Distribution and blue one is our data.

We can get an idea check out Histogram for how much Gauss ( there is Normal Distribution or not). But there is another way to get it called QQ Plot

![Picture8](https://user-images.githubusercontent.com/30235603/78386422-375d2300-75e6-11ea-9c4a-8bdcea82e4ff.png)

You can see the Skewness at tails. I will figure out that with np.log1b (Log Transform)

----

After Log Transform value of Skewness was minimized.

![Picture9](https://user-images.githubusercontent.com/30235603/78386461-47750280-75e6-11ea-8654-b5a7c6da4006.png)

----

I fixed target values. Then I will fix features variables.

![Picture10](https://user-images.githubusercontent.com/30235603/78386481-4e9c1080-75e6-11ea-8d29-82afaeb856a6.png)

Target values has been fixed, Horsepower is so close to 1, I don’t need to fix that

Bu if have to do it I would be using **Box Cox Transformation**. With this method you can fix Skewness values.

---

**2-)** **One Hot Encoding**

Categorical data breaks our model. Therefore I apply Cylinder and Origin Columns.

![Picture11](https://user-images.githubusercontent.com/30235603/78386499-58257880-75e6-11ea-9f21-26370d58b883.png)

Apply that, it is so simple. But first you have to turn categorical data as string. Don’t forget.

----

**Ridge Regression L2:**

It avoids Overfitting and try to minimize this formula

![Picture12](https://user-images.githubusercontent.com/30235603/78386515-5f4c8680-75e6-11ea-9631-8a31af19eae7.png)

**Lasso Regression L1:**

![Picture13](https://user-images.githubusercontent.com/30235603/78386538-64a9d100-75e6-11ea-8d01-d006e52c490f.png)

If COEFs isn’t necessary, Lasso gives 0 ‘em all. Ridge don’t do that. The other best thing about Lasso, if there are high correlated features Lasso use only most important one of them.

**Elastic Net:**

We can say mixture of Lasso and Ridge. It is used for Feature Extraction.

-----
