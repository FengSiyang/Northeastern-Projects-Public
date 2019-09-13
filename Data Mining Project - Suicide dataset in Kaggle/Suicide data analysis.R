rm(list=ls()) # wipe environment

#install.packages("ggplot2")

require(dplyr)
require(tidyr)
require(ggplot2)

# read dataset
df <- read.csv("master.csv")

################# data cleaning #################
# overview of dataset
dim(df)
str(df)
colnames(df)

# standard column name
colnames(df) <- c('country', 'year', 'sex', 'age', 'suicides_no', 'population',
                  'suicide_rate', 'country_year', 'HDI', 'gdp_for_year', 
                  'gdp_per_capita', 'generation')

attach(df)

colnames(df)

## convert data structure
# define the function of converting factor into number
facToNum <- function(column) {
  column <- as.character(column)
  column <- gsub(',', '', column) %>%
    as.numeric()
  return(column)
}

# convert factor of gdp_for_year into numeric
df$gdp_for_year <- facToNum(df$gdp_for_year)

# check NA in each column
anyNA(df)
summary(is.na(df))


# omit HDI column because there are branch of NA value
newdf <- df[-c(9)]
newdf$gender <- sapply(newdf$sex, function(x) {ifelse(x=='male', 1, 2)})

# add column of age with value 1 to 6
ageLabel <- function(x) {
  if (x == '5-14 years') return(1)
  else if (x == '15-24 years') return(2)
  else if (x == '25-34 years') return(3)
  else if (x == '35-54 years') return(4)
  else if (x == '55-74 years') return(5)
  else return(6)
}
newdf$new_age <- sapply(newdf$age, ageLabel)


str(newdf)
summary(is.na(newdf)) # check NA


##################### Exploratory Data Analysis (EDA) and extra data wrangling ################
# total suicide number, per year per country
suicide_overview <- newdf %>%
  group_by(year, country) %>%
  summarise(suiNo_year_country = sum(suicides_no), 
            country_population = sum(population),
            gdp_for_year = mean(gdp_for_year),
            gdp_per_capita = mean(gdp_per_capita)) %>%
  mutate(total_suicide_rate = suiNo_year_country / country_population * 100000) %>%
  ungroup()

# check the recorded country number per year
country_peryear <- suicide_overview %>%
  group_by(year) %>%
  summarise(country_num = n()) %>%
  ungroup()
#check record number with boxplot
boxplot(country_peryear$country_num, 
        main = "Recorded Country Number per Year",
        xlab = "country number",
        col = 'light blue',
        notch = TRUE,
        horizontal = TRUE)
# get the outlier based on the boxplot which is the min value
# outlier: 2016
filter(country_peryear, country_peryear$country_num == min(country_peryear$country_num))


# count the number of recorded countries
country_sum <- suicide_overview %>%
  group_by(country) %>%
  summarise(count_freq = n())
boxplot(country_sum$count_freq, 
        main = "Recorded number for different country",
        xlab = 'recorded number per country',
        col = 'light green',
        notch = TRUE,
        horizontal = TRUE)
# check outliers based on the boxplot
outliers <- filter(country_sum, country_sum$count_freq < 10)
outliers

# cleaning rare country observations
out_country <- c(as.character(outliers$country))
cleaned_df <- filter(newdf, !(newdf$country %in% out_country))



# suicide rate per year
suicide_year <- cleaned_df %>%
  group_by(year) %>%
  summarise(sui_rate_year = sum(suicides_no) / sum(population) * 100000) %>%
  ungroup()

ggplot(suicide_year, aes(suicide_year$year, suicide_year$sui_rate_year)) + 
  geom_line(color = "blue", linetype = "dashed", size = 1.2) + 
  geom_point(size = 3) + 
  ggtitle("Average suicide rate (per 100k population) Per Year")

# cleaning all the observation from 2016
cleaned_df = filter(cleaned_df, cleaned_df$year != 2016)


############################ log transformation and data normalization ########################
#### working for modeling
View(cleaned_df[c(5, 6, 7, 9, 10)])

new_suicide_overview <- cleaned_df %>%
  group_by(year, country) %>%
  summarise(suiNo_year_country = sum(suicides_no), 
            country_population = sum(population),
            gdp_for_year = mean(gdp_for_year),
            gdp_per_capita = mean(gdp_per_capita)) %>%
  mutate(total_suicide_rate = suiNo_year_country / country_population * 100000) %>%
  ungroup()


model_df <- cleaned_df


# normalization (Min-Max normalization)
model_df[c(5, 6, 7, 9, 10)] <- sapply(model_df[c(5, 6, 7, 9, 10)], 
                                      function(x) {
                                        return((x - min(x)) / (max(x) - min(x)))
                                      })

suicide_overview_norm <- new_suicide_overview
suicide_overview_norm[-c(1, 2)] <- sapply(new_suicide_overview[-c(1, 2)], 
                                         function(x) {
                                           return((x - min(x)) / (max(x) - min(x)))
                                         })

##############################################################################################


###### EDA
ggplot(model_df, aes(x = gdp_per_capita, y = suicide_rate, shape = sex, color = sex)) + 
  geom_point() + 
  geom_smooth(method = 'lm', se = FALSE) + 
  ggtitle("GDP per person VS suicide rate (sex identify)")


ggplot(model_df, aes(suicide_rate, color = sex)) +
  geom_density() + 
  ggtitle("Suicide rate density compared with sex")


ggplot(model_df, aes(x = age, y = suicide_rate, fill = sex, color = sex)) +
  geom_point() + 
  ggtitle("Relations between age and suicide rate")


#install.packages("corrplot")
library(corrplot)
corrplot(cor(model_df[-c(1, 3, 4, 8, 11)]), order = 'hclust', tl.col = 'black', tl.cex = .65)

corrplot(cor(suicide_overview_norm[-c(2)]), order = 'hclust', tl.col = 'black', tl.cex = .65)

################################# panel data regression #####################################
#install.packages("foreign")
#install.packages("plm")
require(foreign)
require(plm)
require(gplots)

attach(suicide_overview_norm)

plotmeans(total_suicide_rate ~ year, main = "Heterogeineity across years", data = suicide_overview_norm)

#### model 1: OLS (Ordinary Least Squares) Model
form <- total_suicide_rate ~ country_population + gdp_per_capita + gdp_for_year
poolv <- plm(form, data = suicide_overview_norm, model = 'pooling', index = c("country", "year"))
summary(poolv)

#### model 2: Between estimation
btv <- plm(form, data = suicide_overview_norm, 
              model = 'between',
              index = c("country", "year"))
summary(btv)


#### model 3: First differences estimation
fdv <- plm(form, data = suicide_overview_norm, 
              model = 'fd',
              index = c("country", "year"))
summary(fdv)

#### Model 4: fixed effects model
fixedv <- plm(form, data = suicide_overview_norm, 
              model = 'within',
              index = c("country", "year"))
summary(fixedv)


#### Model 5: Random effects model
randomv <- plm(form, data = suicide_overview_norm, 
               model = 'random',
               index = c("country", "year"))
summary(randomv)

#### model test
# fixed vs random (hausman test)
phtest(randomv, fixedv)   # fixed

# fixed vs OLS (pFtest)
pFtest(fixedv, poolv)   # fixed

fixed.time <- plm(total_suicide_rate ~ country_population + gdp_per_capita + gdp_for_year + factor(year), 
                  data = suicide_overview_norm, 
                  model = 'within',
                  index = c("country", "year"))
summary(fixed.time)
pFtest(fixed.time, fixedv)   # use time-fixed effects (fixed.time)






