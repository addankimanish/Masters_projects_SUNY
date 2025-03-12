#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

#setwd("~/Spring 2022/SDM-2/SDM2 Project")
#Activating packages
library(data.table)
library(ggplot2)
library(cluster)
library(factoextra)
library(dplyr)
library(shiny)
library(shinydashboard)
library(reshape2)


library(tidyr)

library(ggplot2)
library(gridExtra)

#Importing dataset
data= fread('train.csv')
data_1= fread('test.csv')
data = data.frame(subset(data, select = -c(Var_1,Segmentation) ))
#scaled_df =  scale(df)
lapply(data,function(x) { length(which(is.na(x)))})
lapply(data,function(x) { length(which(x==""))})

# There are blank values in categorical variables Graduated and Profession, replacing them with NAs for better handling

data[data==""]<-NA

# Replacing NA rows with mode 
# Creating a User defined function for finding Mode as R does not come with a base mode function
getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Replacing categorical variable NAs with their mode and continuous variable NAs with Mean
data$Ever_Married[is.na(data$Ever_Married)]<-getmode(data$Ever_Married)

data$Graduated[is.na(data$Graduated)]<-getmode(data$Graduated)

data$Profession[is.na(data$Profession)]<-getmode(data$Profession)

data$Work_Experience[is.na(data$Work_Experience)]<-mean(data$Work_Experience,na.rm=TRUE)

data$Family_Size[is.na(data$Family_Size)]<-mean(data$Family_Size,na.rm=TRUE)


# One Hot encoding categorical variables
#install.packages('fastDummies')
library('fastDummies')
cluster_data<-dummy_cols(data)

# Dropping columns not required for analysis

cat <- c("ID","Gender","Ever_Married","Graduated","Profession","Spending_Score")
cluster_data<-cluster_data[ , !(names(cluster_data) %in% cat)]
#knitr::kable(head(cluster_data))

library(cluster)
#install.packages("ClusterR")
library(ClusterR)

WCSS = vector()
for (i in 1:10) {WCSS[i] = sum(kmeans(x=cluster_data,centers = i)$withinss)}



set.seed(240) # Setting seed
kmeans_1 <- kmeans(cluster_data, centers = 4, nstart = 100)
#print(kmeans_1)
data1 = data
data1$segment<-kmeans_1$cluster
#kmeans_1$cluster

#PAM
# Compute PAM
library("cluster")
pam.res <- pam(cluster_data, 4)
data2 = data
data2$segment<-pam.res$cluster



# Shiny Server
server <- shinyServer(function(input,output){
    output$clustdata <- renderTable(data1[1:200,])
    output$summaryDset <- renderPrint({
        #summary(data$informedDset) 
        summary(data[[input$informedDset]]) 
    })
    output$summaryCat <- renderPrint({
        #summary(data$informedDset) 
        table(data[[input$informedCat]])
    }) 
    output$barplot <- renderPlot({
        ggplot(data) + geom_bar(aes(x = Gender,fill = Gender))
    })
    output$histogram1 <- renderPlot({
        ggplot(data) + geom_bar(aes(x = Profession,fill = Profession))
    })
    output$histogram2 <- renderPlot({
        ggplot(data) + geom_bar(aes(x = Spending_Score,fill = Spending_Score))
    })
    output$barplts1 <- renderPlot({
        boxplot(data$Age, horizontal = TRUE,col = 'Purple',main="Age")
        boxplot(data$Work_Experience, horizontal = TRUE,col = 'Orange',main="Work Ex")
        boxplot(data$Family_Size, horizontal = TRUE,col = 'Blue',main="Family Size")
    })
    output$rawdata <- renderTable(data[1:200,])
    output$method1 <- renderPlot({
        fviz_nbclust(cluster_data, kmeans, method = "wss") +
            geom_vline(xintercept = 4, linetype = 2)+
            labs(subtitle = "Elbow method")
    })
    output$method2 <- renderPlot({
        fviz_nbclust(cluster_data, kmeans, method = "silhouette") +
            labs(subtitle = "Silhouette Method")
    })
    output$method3 <- renderPlot({
        set.seed(123)
        fviz_nbclust(cluster_data,kmeans,method = "wss") + 
            labs(subtitle = "Elbow method")
        
    })
    
    output$clusterchart <- renderPlot({
        clusParam = 'kmeans'
        # if(input$ClusterName == 'K-means'){
        #     clusParam = "kmeans"
        # }
        # else if(input$ClusterName == 'PAM'){
        #     clusParam = "pam"
        # }
        # else if(input$ClusterName == 'Hierarchical'){
        #     clusParam = "hclust"
        # }
        # else{
        #     clusParam = "kmeans"
        # }
        fviz_cluster((eclust(cluster_data, clusParam, k = input$clustnum, nstart = 25, graph = FALSE)), geom = "point", ellipse.type = "norm",
                     palette = "jco", ggtheme = theme_minimal())
        
    })
    output$clusterBoxPlot <- renderPlot({
        
        # comp.df <- cbind(cluster_data, kmeans_1$cluster)
        # names(comp.df)[22] <- "cluster"
        # 
        # melted<- melt(comp.df[c(1:3,22)],id.vars="cluster")
        # 
        # ggplot(melted, aes(x = cluster, y = value)) + 
        #     geom_boxplot()+facet_wrap(~variable)+ ggtitle("Box plot of cluster vs NC features")
        # 
        df3 = cbind(cluster_data, kmeans_1$cluster)

        test <- ggplot(df3, aes(group=Age, x = kmeans_1$cluster, y = Age)) +
            geom_boxplot() + theme_bw()
        test1 <- boxplot(data$Age ~ kmeans_1$cluster,
                         xlab='Cluster', ylab='Age',
                         main='Customer Age by Cluster',col=c("steelblue", "darkred", "darkgreen","yellow"),
                         col.bg = "grey")
        
        test2 <- ggplot(df3, aes(group=Age,x = kmeans_1$cluster, y = Work_Experience)) + 
            geom_boxplot() + theme_bw()
        
        test3 <- ggplot(df3, aes(group=Age,x = kmeans_1$cluster, y = Family_Size)) + 
            geom_boxplot() + theme_bw()
        
        grid.arrange(test,test2,test3)
    })

    output$agevswxp_pt <- renderPlot({
        plot(cluster_data[, c("Age", "Work_Experience")], col=kmeans_1$cluster+1, main="K-means")
    })
    output$fsvswxp_pt <- renderPlot({
        plot(cluster_data[, c("Family_Size", "Work_Experience")], col=kmeans_1$cluster+1, main="K-means")
    })
    output$agevsfs_pt <- renderPlot({
        plot(cluster_data[, c( "Age","Family_Size")], col=kmeans_1$cluster+1, main="K-means")
    })
    output$agebpt <- renderPlot({
        boxplot(data$Age ~ kmeans_1$cluster,
                xlab='Cluster', ylab='Age',
                main='Customer Age by Cluster',col=c("steelblue", "darkred", "darkgreen","yellow"),
                col.bg = "grey")
    })
    output$workexbp <- renderPlot({
        boxplot(data$Work_Experience ~ kmeans_1$cluster,
                xlab='Cluster', ylab='Work Experience',
                main='Customer Work Experience by Cluster',col=c("steelblue", "darkred", "darkgreen","yellow"),
                col.bg = "grey")
    })
    output$famsizebpt <- renderPlot({
        boxplot(data$Family_Size ~ kmeans_1$cluster,
                xlab='Cluster', ylab='Family Size',
                main='Customer Family Size by Cluster',col=c("steelblue", "darkred", "darkgreen","yellow"),
                col.bg = "grey")
    })
    
    
    
    
    # #PAM
    output$clusterchart_pam <- renderPlot({
        fviz_cluster((eclust(cluster_data, 'pam', k = input$clustnum_pam, nstart = 25, graph = FALSE)), geom = "point", ellipse.type = "norm",
                     palette = "jco", ggtheme = theme_minimal())

    })
    output$clusterBoxPlot_pam <- renderPlot({

        df3 = cbind(cluster_data, pam.res$cluster)

        test <- ggplot(df3, aes(group=Age, x = pam.res$cluster, y = Age)) +
            geom_boxplot() + theme_bw()
        test1 <- boxplot(data$Age ~ pam.res$cluster,
                         xlab='Cluster', ylab='Age',
                         main='Customer Age by Cluster',col=c("steelblue", "darkred", "darkgreen","yellow"),
                         col.bg = "grey")

        test2 <- ggplot(df3, aes(group=Age,x = pam.res$cluster, y = Work_Experience)) +
            geom_boxplot() + theme_bw()

        test3 <- ggplot(df3, aes(group=Age,x = pam.res$cluster, y = Family_Size)) +
            geom_boxplot() + theme_bw()

        grid.arrange(test,test2,test3)
    })

    output$agebpt_pam <- renderPlot({
        boxplot(data$Age ~ pam.res$cluster,
                xlab='Cluster', ylab='Age',
                main='Customer Age by Cluster',col=c("steelblue", "darkred", "darkgreen","yellow"),
                col.bg = "grey")
    })
    output$workexbp_pam <- renderPlot({
        boxplot(data$Work_Experience ~ pam.res$cluster,
                xlab='Cluster', ylab='Work Experience',
                main='Customer Work Experience by Cluster',col=c("steelblue", "darkred", "darkgreen","yellow"),
                col.bg = "grey")
    })
    output$famsizebpt_pam <- renderPlot({
        boxplot(data$Family_Size ~ pam.res$cluster,
                xlab='Cluster', ylab='Family Size',
                main='Customer Family Size by Cluster',col=c("steelblue", "darkred", "darkgreen","yellow"),
                col.bg = "grey")
    })

    
    
    
    
})















