#This code demonstrates how to create a Broken-Stick plot
#Function was not originally created by Mohan Gupta
#Example produced by Mohan Gupta (https://mohanwugupta.com)
#Github: mohanwugupta
#If you find this tutorial and code useful, please share it with your students and colleagues!

evplot = function(ev) {
  # Broken stick model (MacArthur 1957)
  n = length(ev)
  bsm = data.frame(j=seq(1:n), p=0)
  bsm$p[1] = 1/n
  for (i in 2:n) bsm$p[i] = bsm$p[i-1] + (1/(n + 1 - i))
  bsm$p = 100*bsm$p/n
  # Plot eigenvalues and % of variation for each axis
  op = par(mfrow=c(2,1),omi=c(0.1,0.3,0.1,0.1), mar=c(1, 1, 1, 1))
  barplot(ev, main="Eigenvalues", col="bisque", las=2)
  abline(h=mean(ev), col="red")
  legend("topright", "Average eigenvalue", lwd=1, col=2, bty="n")
  barplot(t(cbind(100*ev/sum(ev), bsm$p[n:1])), beside=TRUE, 
          main="% variation", col=c("bisque",2), las=2)
  legend("topright", c("% eigenvalue", "Broken stick model"), 
         pch=15, col=c("bisque",2), bty="n")
  par(op)
}

#read in data - CHANGE THE PATH - FILE CSV CAN BE FOUND ON MY GITHUB
HCP = read.csv(file="/Users/CCNLAB/Box Sync/Education_Youtube/Code/Machine Learning/HCP_morph_tutorial_dat.csv", header = TRUE)

#remove any NA values
HCP = na.omit(HCP)

#Create PCs
HCP_pca = prcomp(HCP[,4:length(HCP)], center = T, scale. = T)

#Broken-Stick model for selection of # of PCs to keep
ev_HCP = HCP_pca$sdev^2
evplot(ev_HCP)

#On the bottom plot, keep the # of PCs as long as tan bar is higher than the red bars. In this case you would 
#keep four PCs based off of the broken-stick plot. 
