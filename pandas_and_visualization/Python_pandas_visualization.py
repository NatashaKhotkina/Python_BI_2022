#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


# ## read_gff

# In[2]:


def read_gff(file_name, sep='\t', names=['Chromosome', 'Source', 'Type', 'Start', 
            'End', 'Score', 'Strand', 'Phase', 'Attributes'], skiprows=1):
    return pd.read_csv(file_name, sep=sep, names=names, skiprows=skiprows)


# In[3]:


table_1 = read_gff('rrna_annotation.gff')


# In[4]:


table_1.head()


# In[5]:


table_1.dtypes


# In[6]:


table_1['Attributes'] = table_1['Attributes'].str.split(pat='=', n=-1, 
                        expand=True)[1].str.split(pat='_', n=-1, expand=True)[0]


# In[7]:


table_1.head()


# ## rRNA_barplot

# In[8]:


RNA_type = table_1.groupby(['Chromosome', 'Attributes']).agg({'Attributes':'count'})


# In[9]:


RNA_type = RNA_type.rename(columns={'Attributes':'Counts'})


# In[10]:


RNA_type = RNA_type.reset_index()


# In[11]:


RNA_type


# In[12]:


plt.figure(figsize=(10,5))
chart = sns.barplot(data=RNA_type, x="Chromosome", y="Counts", hue="Attributes")
chart.set_xticklabels(chart.get_xticklabels(), rotation=90);


# ## read_bed6

# In[13]:


def read_bed6(file_name, sep='\t', names=['Chromosome', 'Start', 'End', 
            'Contig_info', 'Some_number', 'Strand']):
    return pd.read_csv(file_name, sep=sep, names=names)


# In[14]:


table_2 = read_bed6('alignment.bed')


# In[15]:


table_2.head()


# In[16]:


table_2.dtypes


# ## intersect

# In[17]:


table_intersect = table_1.merge(table_2, how='cross')


# In[18]:


table_intersect = table_intersect.query("Start_x >= Start_y and End_x <= End_y and \
                                        Chromosome_x == Chromosome_y").drop(columns=['Chromosome_y'])


# In[19]:


table_intersect


# ## volcano plot

# In[20]:


diff_expr = pd.read_csv('diffexpr_data.tsv.gz', sep='\t')


# In[21]:


diff_expr.head()


# In[22]:


import math

math.log10(0.05)


# In[23]:


p_treshold = -math.log10(0.05)


# In[24]:


sign_down = diff_expr.query("logFC < 0 and log_pval > @p_treshold")
sign_up = diff_expr.query("logFC > 0 and log_pval > @p_treshold")
nonsign_up = diff_expr.query("logFC > 0 and log_pval <= @p_treshold")
nonsign_down = diff_expr.query("logFC < 0 and log_pval <= @p_treshold")


# In[25]:


top_up = sign_up["logFC"].nlargest(n=2)
genes_up = sign_up.loc[sign_up['logFC'].isin(top_up)]

top_down = sign_down["logFC"].nsmallest(n=2)
genes_down = sign_down.loc[sign_down['logFC'].isin(top_down)]


# In[26]:


SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

fig, ax = plt.subplots(figsize=(20,13), dpi=300, linewidth=10)
#ax.set_facecolor('white')

plt.scatter(sign_down["logFC"], sign_down["log_pval"], label='Significantly downregulated')
plt.scatter(sign_up["logFC"], sign_up["log_pval"], label='Significantly upregulated')
plt.scatter(nonsign_down["logFC"], nonsign_down["log_pval"], label='Non-significantly downregulated')
plt.scatter(nonsign_up["logFC"], nonsign_up["log_pval"], label='Non-significantly upregulated')
plt.axhline(-math.log10(0.05), linestyle='dashed', linewidth=3, color='grey')
plt.axvline(linestyle='dashed', linewidth=3, color='grey')
plt.text(7, 3, "p_value = 0.05", fontsize=20, color='grey', fontweight='bold')

plt.xlabel("log2(fold change)".translate(SUB), size=25, style='italic', fontweight='bold')
plt.ylabel("-log10(p value corrected)".translate(SUB), size=25, style='italic', fontweight='bold')
plt.title("Volcano plot", size=35, style='italic', fontweight='bold')

ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
plt.tick_params(axis='both', which='major', width=3, length=9, labelsize=20)
plt.tick_params(axis='both', which='minor', width=2, length=5, labelsize=8)
limit = max(map(abs,[min(diff_expr["logFC"]), max(diff_expr["logFC"])])) + 1
ax.set_xlim([-limit, limit])

x_ind = genes_down['logFC'].iloc[0]
y_ind = genes_down['log_pval'].iloc[0]
plt.text(x_ind, y_ind+10, genes_down['Sample'].iloc[0], fontsize=20, fontweight='bold')
plt.arrow(x_ind+0.5, y_ind+9, -0.5, -8, width = 0.1, length_includes_head=True, 
          head_length=2, facecolor='red', edgecolor='black')

x_ind = genes_down['logFC'].iloc[1]
y_ind = genes_down['log_pval'].iloc[1]
plt.text(x_ind, y_ind+10, genes_down['Sample'].iloc[1], fontsize=20, fontweight='bold')
plt.arrow(x_ind+0.5, y_ind+9, -0.5, -8, width = 0.1, length_includes_head=True, 
          head_length=2, facecolor='red', edgecolor='black')

x_ind = genes_up['logFC'].iloc[0]
y_ind = genes_up['log_pval'].iloc[0]
plt.text(x_ind+0.5, y_ind+8, genes_up['Sample'].iloc[0], fontsize=20, fontweight='bold')
plt.arrow(x_ind+0.5, y_ind+7, -0.5, -6, width = 0.1, length_includes_head=True, 
          head_length=2, facecolor='red', edgecolor='black')

x_ind = genes_up['logFC'].iloc[1]
y_ind = genes_up['log_pval'].iloc[1]
plt.text(x_ind, y_ind+10, genes_up['Sample'].iloc[1], fontsize=20, fontweight='bold')
plt.arrow(x_ind+0.5, y_ind+9, -0.5, -8, width = 0.1, length_includes_head=True, 
          head_length=2, facecolor='red', edgecolor='black')


[x.set_linewidth(3) for x in ax.spines.values()]
plt.legend(prop={'size': 24, 'weight': 'bold'}, markerscale=3, shadow=True);


# ## Covid

# In[27]:


covid = pd.read_csv("owid-covid-data.csv")


# In[28]:


covid.columns


# In[29]:


covid.head()


# In[30]:


# Find the latest date to see the latest statistics
dates = covid['date'].str.split(pat="/", expand=True).rename(columns = {0:'Month', 1: 'Day', 2: 'Year'})
dates.sort_values(['Year', 'Month', 'Day'], ascending=False, inplace=True)
dates.head()


# In[31]:


latest_data = covid.query('date == "9/9/2022"')


# In[32]:


sns.set(rc = {'figure.figsize':(30,16)})
graph = sns.boxplot(x=latest_data['continent'], y=latest_data['total_cases_per_million'])
#graph.axes.set_title("Title",fontsize=50)
graph.set_xlabel("Continent",fontsize=40)
graph.set_ylabel("Total cases per million",fontsize=40)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
#graph.tick_params(labelsize=5)
plt.show()


# In Europe there were relatively a lot of cases and in Africa relatively few cases of covid per million.

# In[33]:


graph = sns.boxplot(x=latest_data['continent'], y=latest_data['total_deaths_per_million'])
#sns.set(rc = {'figure.figsize':(30,16)})
#graph.axes.set_title("Title",fontsize=50)
graph.set_xlabel("Continent",fontsize=40)
graph.set_ylabel("Total deaths per million",fontsize=40)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
#graph.tick_params(labelsize=5)
plt.show()


# In Europe there were also quite a lot of deaths per million. Interestingly, there were the same amount of cases per million in North and in South America. But there were more deaths in South America.

# In[34]:


#Let's find out whether number of vaccinated people is correlated with number of deaths per million
latest_data['log_vac'] = np.log10(latest_data['people_vaccinated_per_hundred'])


# In[35]:


sns.lineplot(x='log_vac', y='total_deaths_per_million', data=latest_data)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.xlabel('Log10(Vacinated per hundred)', size=40)
plt.ylabel('Total death per million', size=40)


# Looks like it makes no sense to get vaccinared...

# In[36]:


latest_data['log10_smoker_per_million'] = np.log10((latest_data['female_smokers'] + latest_data['male_smokers']) 
                                     / latest_data['population'] * 10**6)


# In[37]:


sns.lineplot(x='log10_smoker_per_million', y='total_cases_per_million', data=latest_data)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.xlabel('Log10(smokers per million)', size=40)
plt.ylabel('Total cases per million', size=40)


# Looks like number of cases slightly increases with increasing number of smokers.

# In[38]:


# Let's check out the dinamic of new cases.
fig = sns.lineplot(x="date", y="new_cases", hue="continent", data=covid, ci=None, linewidth=3, 
                   palette=["black", "forestgreen", "deeppink", "royalblue", "orange", 'grey'])
plt.xticks(ticks=range(0, 1001, 100), rotation=30)
plt.ylim(0)
plt.xlabel('Date', size=40)
plt.ylabel('New cases', size=40)
leg = plt.legend(prop={'size': 24})
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)

for line in leg.get_lines():
    line.set_linewidth(8.0)


# Looks like in the beginning there were a lot of new cases in South America, and not in Asia or Europe. In the end of 2021 and beginning of 2022 there was a peak everywhere.
