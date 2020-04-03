In this project I worked on [Auto-MPG Dataset](https://www.kaggle.com/uciml/autompg-dataset) with Advanced Machine Learning techniques.

During the project, I applied different ML techniques likes Ridge Regression, Losso Regression, Outlier Detection, etc.
I am going to explain something about code at below

**Why did I define something called threshold?**

The purpose is more visibility on pair plot and other purpose is minimize the correlation matrix.

[IMAGES]

By the way if there are more correlated features each other, we can say they are multicollinearity. This situation is disadvantage because we use more features instead of one feature.

----

When the pair plot is drawn, it looks like there are Outliers:

[IMAGE]

-----
[IMAGE]

After Box Plot, I am really sure now, Horsepower and Acceleration have outliers.

**And then I find the outliers:**

[IMAGE]

After finding Outliers, as you can see, shape is reduced. I will apply these codes for acceleration. After all that, last shape is **(392,8)**
It means totally 6 Outliers has been removed with these filters.

**Feature Engineering**

**1-)** **Skewness**

[IMAGE]

3rd momentum of data gives value of Skewness. I will use scipy library for calculating that. (Skewness > 1 -> positive, Skewness < -1 -> negative)

Also Skewness is named Outliers of our data in tail field.

Using Log Transform we make a transformation from high slope dispersion to low slope dispersion, that mean we can minimize value of skew.

Before Log Transform you can see the Skewness.

[IMAGE]

Black one is Normal Distribution and blue one is our data.

We can get an idea check out Histogram for how much Gauss ( there is Normal Distribution or not). But there is another way to get it called QQ Plot

[IMAGE]

You can see the Skewness at tails. I will figure out that with np.log1b (Log Transform)


After Log Transform value of Skewness was minimized.

[IMAGE]

I fixed target values. Then I will fix features variables.
[IMAGE]
Target values has been fixed, Horsepower is so close to 1, I don’t need to fix that

Bu if have to do it I would be using **Box Cox Transformation****.** With this method you can fix Skewness values.

---

**2-)** **One Hot Encoding**

Categorical data breaks our model. Therefore I apply Cylinder and Origin Columns.

[IMAGE]

Apply that, it is so simple. But first you have to turn categorical data as string. Don’t forget.

----

**Ridge Regression L2:**

It avoids Overfitting and try to minimize this formula

[IMAGE]

**Lasso Regression L1:**

[IMAGE]

If COEFs isn’t necessary, Lasso gives 0 ‘em all. Ridge don’t do that. The other best thing about Lasso, if there are high correlated features Lasso use only most important one of them.

**Elastic Net:**

We can say mixture of Lasso and Ridge. It is used for Feature Extraction.