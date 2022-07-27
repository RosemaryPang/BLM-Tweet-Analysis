##### R's command line
# Doing math in R
1+3
(1+5+23+14+64)/5

# Sometimes R shows an error message
2^^2


# Assign names to values
x <- 2 
y <- x^2 + 2  #assignment does not print output
y 

# We can redefine values
x <- "two"
x

# We can assign more than one numbers to a variable
x <- c(89, 12, 75, 103, 325)
x
x+x

# Built-in functions
sqrt(4)
log(100)   #log base e = exp(1)
log(100,10) #log base 10
length(x) #the number of values in a variable
sum(x) #adding all the values in a variable

##### The workspace
ls() #lists the currently defined objects in the workspace
rm(x) #remove single object from workspace
rm(list=ls()) #remove all objects


##### External packages
install.packages("janitor")
library(janitor)


##### Get help
apropos('mean') #List objects whose names match "mean"

help("mean") #Find help on the mean function
?mean

example("mean") #Run examples found in help page for mean

help(package="janitor")
