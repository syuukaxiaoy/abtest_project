
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 

# <a id='probability'></a>
# #### Part I - Probability
# 

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head(10)


# b. Use the below cell to find the number of rows in the dataset.

# In[3]:


df.shape


# c. The number of unique users in the dataset.

# In[4]:


df['user_id'].nunique()


# d. The proportion of users converted.

# In[5]:


df[df['converted'] == 1].user_id.count()/df['user_id'].count()


# e. The number of times the `new_page` and `treatment` don't line up.

# In[6]:


df_t = df.query('group == "treatment"')
df_c = df.query('group == "control"')
n_mis = df_t.query('landing_page == "old_page"').user_id.count() + df_c.query('landing_page == "new_page"').user_id.count()

n_mis


# f. Do any of the rows have missing values?

# In[7]:


df.info()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, 
#we cannot be sure if this row truly received the new or old page. 
# 
# a. Now use the answer to the quiz to create a new dataset.  
# In[8]:


df_t_o = df_t.query('landing_page == "old_page"')
df_c_n = df_c.query('landing_page == "new_page"')
mis = pd.concat([df_t_o,df_c_n])

df2 = df
mis_index = mis.index
df2 = df2.drop(mis_index)


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells 

# a. How many unique **user_id**s are in **df2**?

# In[10]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


df2[df2.duplicated('user_id')]


# c. What is the row information for the repeat **user_id**? 

# In[12]:


df2.query('user_id == 773192')


# d. Remove **one** of the rows with a duplicate **user_id**, but keep dataframe as **df2**.

# In[13]:


df2.drop(labels=1899,axis=0,inplace=True)


# In[14]:


df2.query('user_id == 773192')



# a. What is the probability of an individual converting regardless of the page they receive?

# In[15]:


df2.query('converted == 1').user_id.count()/df2.user_id.count()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[16]:


df2_c = df2.query('group == "control"')
df2_c.query('converted == 1').user_id.count()/df2_c.user_id.count()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[17]:


df2_t = df2.query('group == "treatment"')
df2_t.query('converted == 1').user_id.count()/df2_t.user_id.count()


# d. What is the probability that an individual received the new page?

# In[18]:


df2.query('landing_page == "new_page"').user_id.count()/df2.user_id.count()


# **The rate of the control group (old page) is higher than the teatment group (new page). But, the difference is just roughly 0.2%.
# From the data , the probability that an individual recieved a new page is roughly 50%, this means that it is not possible to be a big difference in conversion with being given more opportunities. **

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# Null-Hypothesis: 
# 
# **$H_{0}$**: **$p_{old}$** >= **$p_{new}$**
# 
# Alternative-hypothesis:
# 
# **$H_{1}$**: **$p_{old}$** < **$p_{new}$**

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[19]:


p_new = df2.query('landing_page == "new_page"').converted.mean()
p_old = df2.query('landing_page == "old_page"').converted.mean()
p_mean = np.mean([p_new, p_old])
p_mean


# In[20]:


p_n0 = p_mean
p_o0 = p_mean


# **$H_{0}$**:p_n0 = 0.1196

# In[21]:


df2.head()


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# **$H_{0}$**:p_o0 = 0.1196

# c. What is $n_{new}$?

# In[22]:


n_new = df2.query('landing_page == "new_page"').user_id.count()
n_new


# d. What is $n_{old}$?

# In[23]:


n_old = df2.query('landing_page == "old_page"').user_id.count()
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[24]:


new_page_converted = np.random.choice([1,0],size=n_new,p=[p_mean,(1-p_mean)])
new_page_converted.mean()


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[25]:


old_page_converted = np.random.choice([1,0],size=n_old,p=[p_mean,(1-p_mean)])
old_page_converted.mean()


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[26]:


p_diff = new_page_converted.mean() - old_page_converted.mean()
p_diff


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[27]:


p_diffs = []
for _ in range(10000):
    b_sample = df2.sample(df2.shape[0],replace=True)
    new_page_converted = np.random.choice([1,0],size=n_new,p=[p_mean,(1-p_mean)])
    old_page_converted = np.random.choice([1,0],size=n_old,p=[p_mean,(1-p_mean)])
    p_diff = new_page_converted.mean() - old_page_converted.mean()
    p_diffs.append(p_diff)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[28]:


p_diff = p_new-p_old


# In[29]:


p_diffs = np.array(p_diffs)
plt.hist(p_diffs);
plt.axvline(x=p_new-p_old,c='r',label="real diff")
plt.axvline(x=(p_diffs.mean()),c='y',label="simulated diff")


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[30]:


(p_diffs > p_diff).mean()


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **It is p-value.If the sample confored to null-hypothesis,the expected proportion is greater than the real-diff to be 0.5.But the 0.9 of the population in simulated sample is greater than the real_diff.We can find that there is no evidence to reject the null hypothesis**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[31]:


old = df2.query('landing_page == "old_page"')
new = df2.query('landing_page == "new_page"')


# In[32]:


import statsmodels.api as sm

convert_old = old.query('converted == 1').user_id.count()
convert_new = new.query('converted == 1').user_id.count()
n_old = df2.query('landing_page == "old_page"').user_id.count()
n_new = df2.query('landing_page == "new_page"').user_id.count()


# In[33]:


convert_old, convert_new, n_old, n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[36]:


z_score, p_value = sm.stats.proportions_ztest(count=[convert_new,convert_old], nobs=[n_new,n_old],alternative='larger')


# In[37]:


(z_score, p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **From the histogram we can say that there is about -1.31 standard deviations between the lines,and the p-value is about 0.905.It's close to j.So that we can't reject the null-hypothesis.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic regression.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[38]:


df3 = df2
df3.head()


# In[39]:


df3['intercept']=pd.Series(np.zeros(len(df3)),index=df3.index)
df3['ab_page']=pd.Series(np.zeros(len(df3)),index=df3.index)
df3.head()


# In[40]:


#creat new columns
row = df3.query('group == "treatment"').index
df3.set_value(index=df3.index, col='intercept', value=1)
df3.set_value(index=row, col='ab_page', value=1)
#change type
df3[['intercept','ab_page']] = df3[['intercept','ab_page']].astype(int)


# In[41]:


df3.query('group == "treatment"').head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[42]:


logit = sm.Logit(df3['converted'],df3[['ab_page','intercept']])
result = logit.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[43]:


result.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **The p-value associated with ab_page is 0.190.
# This is because Part II incorrectly used a two-tailed test,leading to the same direction of the two inspections.If the "one-tailed test" is correctly used in Part II, it should be different.
#  We added an intercept which is meant to let the p-value is more accurate.But, this p-value is still much too high to reject the null hypothesis.**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **I consider whether the users'duration on the page, user gender, age, or the place of residence will affect the convertion rate.The men &women ,or different age,area of user may be a little biased on the version of page.And from analysis of every users' duration on page ,we can also know whether the user willing to upgreat the version. But it's not advisable to add many factors.Because sometimes in regression analysis, the kind of factor and the expression of this factor is only a kind of speculation, the intestability and diversity  make regression analysis restricted in some cases.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[44]:


df_countries = pd.read_csv('countries.csv')
df_countries.country.unique()


# In[45]:


df_dummies = pd.get_dummies(df_countries,columns=['country'])

df4 = df3.merge(df_dummies,on='user_id')
df4.head()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[46]:


logit = sm.Logit(df4['converted'],df4[['intercept','country_CA','country_UK']])
result = logit.fit()
result.summary()


# **From result we can say that it seems having a certain influence on the conversion rate,but it is not high enough.The next we add page&country to see weather it influence on conversion rate.**

# In[47]:


df4['page_CA'] = df4['ab_page'] * df4['country_CA']
df4['page_UK'] = df4['ab_page'] * df4['country_UK']


# In[50]:


logit1 = sm.Logit(df4['converted'],df4[['intercept', 'country_CA','country_UK', 'ab_page', 'page_CA','page_UK']])
result1 = logit1.fit()
result1.summary()


# **It seems that the p-values for all columns have been increased when we added each factor.The z-score of the intercept is also very large.**

# Comment：
# 
# 1.At first it looks like there is a difference in conversion rates between old and new page, but there is not enough evidence to reject the null hypothesis. From the analysis shown in this report, the new page is not better than the old version.
# 
# 2.We also found that users have approximately 50% chance of receiving old or new page. But it doesn't depend on countries with approximately the same conversion rates as US or UK.
# 
# 3.Although we used page&country, it seems having a certain influence on the conversion rate, but it is not high enough.The z-score of the intercept is also very large. So I am thinking about the other reasons,for example the quality ,design of the webside.

# <a id='conclusions'></a>
# ## Conclusions
# 
# Congratulations on completing the project! 
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by continuing on to the next module in the program.
