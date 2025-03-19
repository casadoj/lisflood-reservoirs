library(bomWater)

# Define the base path for saving data
base_path <- "Z:/nahaUsers/casadje/datasets/reservoirs/ResOpsAU/raw/"

# period of extraction
start_date <- '1975-01-01'
end_date <- as.character(Sys.Date())

# Create the subfolders if they don't exist
timeseries_folder <- file.path(base_path, "time_series")
attributes_folder <- file.path(base_path, "attributes")

# Create directories if they don't exist
if (!dir.exists(timeseries_folder)) {
  dir.create(timeseries_folder, recursive = TRUE)
}
if (!dir.exists(attributes_folder)) {
  dir.create(attributes_folder, recursive = TRUE)
}


# loop over parameters
parameters <- c('Storage Volume', 'Storage Level')
for (parameter_type in parameters) {
  
  par_name = tolower(strsplit(parameter_type, " ")[[1]][2])
  
  # Get the station list
  stations = get_station_list(parameter_type = parameter_type)
  
  # Save the stations table as a CSV file in the "attributes" folder
  file_name <- paste0("stations_", par_name, ".csv")
  write.csv(stations, file = file.path(attributes_folder, file_name), row.names = FALSE)
  
  # Loop over station numbers
  for (station in stations$station_no) {
    
    tryCatch({
      
      # Extract the time series for each station
      timeseries <- get_daily(parameter_type = parameter_type,
                              station_number = as.character(station),
                              start_date     = start_date,
                              end_date       = end_date)
      
      # Check if volume data is available
      if (!is.null(timeseries)) {
        
        # Export the time series to a CSV file in the "timeseries" folder
        file_name <- paste0(station, "_", par_name, ".csv")
        write.csv(timeseries, file = file.path(timeseries_folder, file_name), row.names = FALSE)
        
        # Optionally, print a message for progress
        cat("Exported time series of", par_name, "for station", station, "\n")
      } else {
        # Print message if no data is available
        cat("No data available for station", station, "\n")
      }
    }, error = function(e) {
      # Handle the error by printing a message and continuing the loop
      cat("Error processing station", station, ": ", e$message, "\n")
    })
  }
}






