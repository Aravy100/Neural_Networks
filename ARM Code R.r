
```{r}
# Loading and/or installing libraries
library("arules")
library("stringr")
library("tidyverse")
```


```{r}
# Read the csv that was created in Python
data <- read.csv('C:/Users/17207/My Drive/From DropBox/00 Data Science (CU)/Spring 2023/Text Mining/data_final.csv')

# Save the label in a separate dataframe
labels <- subset(data, select="Label")

labels <- labels %>% mutate(Label=str_remove_all(Label, " cuisine"))

# Drop everything except for the content column
df <- subset(data, select="content")

# Split the single column into a transaction type dataframe
df <- str_split_fixed(df$content," ",n=Inf)

# Save the document as a CSV with the labels
write.csv(df,'C:/Users/17207/My Drive/From DropBox/00 Data Science (CU)/Spring 2023/Text Mining/df_transactions_input_sanslabel.csv', row.names = FALSE)

# Add the label column at the beginning
df <- cbind(labels,df)

# Save the document as a CSV with the labels
write.csv(df,'C:/Users/17207/My Drive/From DropBox/00 Data Science (CU)/Spring 2023/Text Mining/df_transactions_input.csv', row.names = FALSE)
```


# There are two dataframes below - one with the label and one without the label
# Dataframe without the Label
```{r}
# Overall Rules
# Read in the saved csv as a transaction object
df_transactions <- read.transactions('C:/Users/17207/My Drive/From DropBox/00 Data Science (CU)/Spring 2023/Text Mining/df_transactions_input_sanslabel.csv',rm.duplicates = FALSE, format = "basket", sep = ",", header=TRUE)

# Run the apriori algorithm and save the results
ARMDF <- arules::apriori(df_transactions, parameter=list(support=0.01,confidence=0.2,minlen=2))

ARMDF <- sort(ARMDF, decreasing=TRUE, by="support")
itemsets <- generatingItemsets(ARMDF)
dupes <- which(duplicated(itemsets))
ARMDF = ARMDF[-dupes]
inspect(ARMDF[1:10])
```



# Soring the above dataframe by Confidence and Lift
```{r}
# Read in the saved csv as a transaction object
df_transactions <- read.transactions('C:/Users/17207/My Drive/From DropBox/00 Data Science (CU)/Spring 2023/Text Mining/df_transactions_input_sanslabel.csv',rm.duplicates = FALSE, format = "basket", sep = ",", header=TRUE)

# Run the apriori algorithm and save the results
ARMDF <- arules::apriori(df_transactions, parameter=list(support=0.01,confidence=0.2,minlen=2))

ARMDF <- sort(ARMDF, decreasing=TRUE, by="confidence")
itemsets <- generatingItemsets(ARMDF)
dupes <- which(duplicated(itemsets))
ARMDF = ARMDF[-dupes]
inspect(ARMDF[1:10])
```

```{r}
itemFrequencyPlot(df_transactions,topN = 25)
```



```{r}
# Read in the saved csv as a transaction object
df_transactions <- read.transactions('C:/Users/17207/My Drive/From DropBox/00 Data Science (CU)/Spring 2023/Text Mining/df_transactions_input_sanslabel.csv',rm.duplicates = FALSE, format = "basket", sep = ",", header=TRUE)

# Run the apriori algorithm and save the results
ARMDF <- arules::apriori(df_transactions, parameter=list(support=0.01,confidence=0.2,minlen=2))

ARMDF <- sort(ARMDF, decreasing=TRUE, by="lift")
itemsets <- generatingItemsets(ARMDF)
dupes <- which(duplicated(itemsets))
ARMDF = ARMDF[-dupes]
inspect(ARMDF[1:10])
```



```{r}
subrules <- head(sort(ARMDF, by="support"),12)
plot(subrules, method="graph", engine="htmlwidget")
```



```{r}
# Set LHS = Chinese, French, etc
# Read in the saved csv as a transaction object
df_transactions <- read.transactions('C:/Users/17207/My Drive/From DropBox/00 Data Science (CU)/Spring 2023/Text Mining/df_transactions_input.csv',rm.duplicates = FALSE, format = "basket", sep = ",", header=TRUE)

# Run the apriori algorithm and save the results
ARMDF <- arules::apriori(df_transactions, parameter =list(support=0.01,confidence=0.02,minlen=2), appearance = list(default="rhs", lhs="chinese"),)
ARMDF <- sort(ARMDF, decreasing=TRUE, by="support")
# Get the top n rules
ARMDF_Cuisines <- inspect(ARMDF[1:5])

ARMDF <- arules::apriori(df_transactions, parameter =list(support=0.01,confidence=0.02,minlen=2), appearance = list(default="rhs", lhs="french"),)
ARMDF <- sort(ARMDF, decreasing=TRUE, by="support")
# Get the top n rules
ARMDF_Cuisines <- rbind(ARMDF_Cuisines,inspect(ARMDF[1:5]))

ARMDF <- arules::apriori(df_transactions, parameter =list(support=0.01,confidence=0.02,minlen=2), appearance = list(default="rhs", lhs="indian"),)
ARMDF <- sort(ARMDF, decreasing=TRUE, by="support")
# Get the top n rules
ARMDF_Cuisines <- rbind(ARMDF_Cuisines,inspect(ARMDF[1:5]))

ARMDF <- arules::apriori(df_transactions, parameter =list(support=0.01,confidence=0.02,minlen=2), appearance =
                           list(default="rhs", lhs="mediterranean"),)
ARMDF <- sort(ARMDF, decreasing=TRUE, by="support")
# Get the top n rules
ARMDF_Cuisines <- rbind(ARMDF_Cuisines,inspect(ARMDF[1:5]))

ARMDF <- arules::apriori(df_transactions, parameter =list(support=0.01,confidence=0.02,minlen=2), appearance = list(default="rhs", lhs="italian"),)
ARMDF <- sort(ARMDF, decreasing=TRUE, by="support")
# Get the top n rules
ARMDF_Cuisines <- rbind(ARMDF_Cuisines,inspect(ARMDF[1:5]))
```