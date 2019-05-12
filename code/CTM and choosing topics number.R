
#install.packages("topicmodels")
#install.packages("ldatuning")
#install.packages("tidytext")
#install.packages("tidyverse")
#install.packages("topicmodels")
#install.packages("stm")
#install.packages("quanteda")
#install.packages("Matrix")
#install.packages("Rtsne")
#install.packages("igraph")

# R version 3.5.1

library(topicmodels)
library(magrittr)  
library(tidyverse) 
library(tidytext) 
library(ldatuning)
library(stm)
library(quanteda)
library(data.table)
library(Rtsne)

# Formatting data ####
testt = read.table("Documents/corpus.csv",sep = ",", header = T,row.names = 1) # Read corpus
testt$X0 <- testt$X0 %>% as.character() 
e <- corpus(x = testt,text_field = "X0",docid_field = "index") # Perform 

# Search for optimal topics number ####
knum <- searchK(dfm(e),K = (c(10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25)),
                N = 112) # Choose topics number
knum %>% plot.searchK()

# Train model ####
stm12 <- stm(dfm(e),K = 12) # Train for k=12
stm15 <- stm(dfm(e),K = 14) # Train for k=15

# Write outputs ####
write_csv(stm12$beta %>% as.data.frame(),"Documents/CTM_12_e.csv")
write_csv(stm12$theta %>% as.data.frame(),"Documents/CTM_12_pi.csv")
write_csv(stm15$beta %>% as.data.frame(),"Documents/CTM_14_e.csv")
write_csv(stm15$theta %>% as.data.frame(),"Documents/CTM_14_pi.csv")

# Visualize topics correlations ####
tpc12 <- topicCorr(stm12)
tpc12$cor %>% heatmap() # Topics correlation heatmap.
tpc12 %>% plot.topicCorr() # Topics neighber graph

tpc15 <- topicCorr(stm15)
tpc15$cor %>% heatmap() 
tpc15 %>% plot.topicCorr() 

# Clustering by dimensionality reduction####
# PCA ####
pca_15 <- prcomp(x = stm15$theta,scale. = F,center = T)
pc1 = (pca_15$sdev / sum(pca_15$sdev) * 100)[1] %>% as.character()
pc2 = (pca_15$sdev / sum(pca_15$sdev) * 100)[2] %>% as.character()
plot(pca_15$x[,c(1:2)],
     xlab = paste(c("PC1 - ", pc1, "%"),collapse = ""),
     ylab = paste(c("PC2 - ", pc2, "%"),collapse = ""))

pca_12 <- prcomp(x = stm12$theta,scale. = F,center = T)
pc1 = (pca_12$sdev / sum(pca_12$sdev) * 100)[1] %>% as.character()
pc2 = (pca_12$sdev / sum(pca_12$sdev) * 100)[2] %>% as.character()
plot(pca_12$x[,c(1:2)],
     xlab = paste(c("PC1 - ", pc1, "%"),collapse = ""),
     ylab = paste(c("PC2 - ", pc2, "%"),collapse = ""))

# Tsne ####
tsne15 <- Rtsne(X = stm15$theta,dims=2,pca=T)
tsne15$Y %>% plot(xlab = "X",ylab = "Y")
tsne12 <- Rtsne(X = stm12$theta,dims=2,pca=T)
tsne12$Y %>% plot(xlab = "X",ylab = "Y")

abline(lm(pca_15$x[,2]~tsne15$Y[,2]), col="blue")
lines(lowess(pca_15$x[,2]~tsne15$Y[,2]), col="red")
plot(tsne15$Y[,2],pca_15$x[,2],method = "pearson",main = "Scatterplot correlation: -0.9249473",xlab = "PCA",ylab = "tsne")

cor(tsne15$Y[,2],pca_15$x[,2])

?lowess
