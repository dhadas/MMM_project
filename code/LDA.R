
library(topicmodels)
library(magrittr) 
library(tidytext)  
library(topicmodels) 

df.train = read.table("data.csv",sep = ",", header = TRUE, row.names = 1)
dtm.train <- df.train %>% cast_dtm(term = ind,document = variable,value = value)

control_list_gibbs <- list(
  burnin = 2500,
  iter = 5000,
  seed = 0:4,
  nstart = 5,
  best = TRUE
)
result_15 <- LDA(k=15, x = dtm.train,method="Gibbs",control =control_list_gibbs)
result.post <- posterior(result_15,dtm.train)
write.csv(x = result.post$terms,file = "LDA_15_term_matrix.csv")
write.csv(x = result.post$topics,file = "LDA_15_topics_matrix.csv")

result_12 <- LDA(k=12, x = dtm.train,method="Gibbs",control =control_list_gibbs)
result.post <- posterior(result_12,dtm.train)
write.csv(x = result.post$terms,file = "LDA_15_term_matrix.csv")
write.csv(x = result.post$topics,file = "LDA_15_topics_matrix.csv")


