library(gstat)
library(dplyr)

raw_data = data(wind)

# Need rows as time and columns as locations this time around

just_vals = wind %>% select(-c(year, month, day))

write.csv(just_vals, file = "Irish_wind_t.csv", row.names = FALSE)
write.csv(wind.loc, file = "Irish_wind_locs.csv", row.names = FALSE)

