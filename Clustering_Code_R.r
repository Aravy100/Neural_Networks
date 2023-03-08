# Clustering - Hierarchical

# Create the corpus manually and now load them
```{r}
SmallCorpus <- Corpus(DirSource("C:/Users/17207/My Drive/From DropBox/00 Data Science (CU)/Spring 2023/Text Mining/cuisines_corpus"))
ndocs<-length(SmallCorpus)

## Do some clean-up.............
SmallCorpus <- tm_map(SmallCorpus, content_transformer(tolower))
SmallCorpus <- tm_map(SmallCorpus, removePunctuation)
## Remove all Stop Words
SmallCorpus <- tm_map(SmallCorpus, removeWords, stopwords("english"))
```




```{r}
# install.packages("tm")
library(tm)

# Convert the data into to a Document Term Matrix
# hclust_in_data<-data$content

SmallCorpus_DTM <- DocumentTermMatrix(SmallCorpus,
                                 control = list(
                                   stopwords = TRUE, ## remove normal stopwords
                                   wordLengths=c(3, 10), ## get rid of words of len 2 or smaller or larger than15
                                   removePunctuation = TRUE,
                                   removeNumbers = TRUE,
                                   tolower=TRUE
                                 ))

inspect(SmallCorpus_DTM)
```




```{r}
# install.packages("NbClust")
library(NbClust)
library(factoextra)
# Convert to DF
SmallCorpus_DF_DT <- as.data.frame(as.matrix(SmallCorpus_DTM))

# Using Sihouette to determine the optimal number of clusters
fviz_nbclust(SmallCorpus_DF_DT, method = "silhouette", FUN = hcut, k.max = 9)

#Source: http://www.sthda.com/english/articles/29-cluster-validation-essentials/96-determiningthe-optimal-number-of-clusters-3-must-know-methods/#:~:text=fviz_nbclust()%20function%20%5Bin%20factoextra,)%2C%20CLARA%2C%20HCUT%5D
```



```{r}
Dist_CorpusM2 <- dist(SmallCorpus_DF_DT, method = "euclidean")

HClust_SmallCorp <- hclust(Dist_CorpusM2, method = "average" )

plot(HClust_SmallCorp)

#plot(Dist_CorpusM2, cex=0.9, hang=-1, main = "Minkowski p=2 (Euclidean)")

rect.hclust(HClust_SmallCorp, k=2)
```


```{r}
# Cosine Similarity Based Denodrogram
(My_m <- (as.matrix(scale(t(SmallCorpus_DF_DT)))))
# Calculating cosine similarity using the actual mathematical equation
(My_cosine_dist = 1-crossprod(My_m) /(sqrt(colSums(My_m^2)%*%t(colSums(My_m^2)))))
# create distance matrix based on cosine
My_cosine_dist <- as.dist(My_cosine_dist)

HClust_Ward_CosSim_SmallCorp2 <- hclust(My_cosine_dist, method="ward.D")
plot(HClust_Ward_CosSim_SmallCorp2, cex=.7, hang=-30,main = "Cosine Sim")
rect.hclust(HClust_Ward_CosSim_SmallCorp2, k=3)
```