library(devtools)

install.packages("httpuv")

devtools::install_github("bleutner/RStoolbox")
remove.packages("RStoolbox")
pkgbuild::check_build_tools(debug = TRUE)
#install_github("bleutner/RStoolbox")
devtools::install_github("16EAGLE/getSpatialData")
remove.packages("htmlwidgets")
install.packages("htmlwidgets")
remove.packages("vctrs")
#load library
library(sf)
library(getSpatialData)
library(terra)

#set main directory
outDir <- file.path("C:/Users/alyyl132/OneDrive - The University of Nottingham/Dissertation/Data/example") #set destination path

#load dam shp dataset
ad<-read_sf("P:/Baleh_project/shp/available_date.shp")[14196,] #change line number according to specifc dam

##buffer to create a polygon
ad_32648<-st_transform(ad,32648)
ad.buff <-st_buffer(ad_32648, 100)

##log in to repositories
login_CopHub(username = "tbox")  # Asdzxc123@
#> Login successfull. ESA Copernicus credentials have been saved for the current session.
login_USGS(username = "TBOX") #!uJf6k3hEGR_s@3
#> Login successfull. USGS ERS credentials have been saved for the current session.
login_earthdata(username = "tbox")  #Asdzxc123@

##set polygon as area of interest
sfc.n<-st_as_sfc(ad.buff)
set_aoi(sfc.n)
view_aoi()

##set a date range
ad_32648$date.time<-substr(ad_32648$Date, start = 1, stop = 19)
ad_32648$date.time<-as.POSIXct(ad_32648$date.time,format="%Y-%m-%d %T")
ad_32648$date.time_min<-ad_32648$date.time-(20*60*60*24)#change number of day if smaller or wider range is needed(first number)
ad_32648$date.time_max<-ad_32648$date.time+(20*60*60*24)#change number of day if smaller or wider range is needed(first number)
ad_32648$min<-substr(as.character(ad_32648$date.time_min), start = 1, stop = 10)
ad_32648$max<-substr(as.character(ad_32648$date.time_max), start = 1, stop = 10)


##find desired products
records <- get_records(time_range = c(ad_32648$min,ad_32648$max),products = c("sentinel-1"))
view_records(records)
records<- check_availability(records)    #this command add rwo coulm at the records sf object. If the needed data are not available for dowload run next comands (now off) 


sent2 <- order_data(records[2,],wait_to_complete = T,verbose = T)#to run if data not available
sent1<- order_data(records[7,],wait_to_complete = T,verbose = T)#to run if data not available

#download data
sent2<- get_data(records[2,],dir_out = outDir)#change number of the desired line. For sentinel 2: lower cloud coverage percentage (cloudcov_highprob column), closest possible to targeted date-time
sent1<- get_data(records[7,],dir_out = outDir)#change number of the desired line. For sentinel 1:  GRD data only, closest possible to targeted date-time and to sentinel 2 data above selected

#unzip sentinel 2 data
unzip(paste0("C:\\Users\\alyyl132\\Downloads\\example\\sentinel-2\\",sent2$record_id,".zip"),exdir = "C:\\Users\\alyyl132\\Downloads\\example\\sentinel-2")

#create a multi layer raster from Sentinel 2 data with R,G,B and NIR bands (10 m resolution)
p2<-list.files(paste0(outDir,"/sentinel-2/",sent2$record_id,".SAFE/GRANULE/"))
i2<-list.files(paste0(outDir,"/sentinel-2/",sent2$record_id,".SAFE/GRANULE/",p2,"/IMG_DATA"))

b2<-rast(paste0(outDir,"/sentinel-2/",sent2$record_id,".SAFE/GRANULE/",p2,"/IMG_DATA/",i2[2]))
b3<-rast(paste0(outDir,"/sentinel-2/",sent2$record_id,".SAFE/GRANULE/",p2,"/IMG_DATA/",i2[3]))
b4<-rast(paste0(outDir,"/sentinel-2/",sent2$record_id,".SAFE/GRANULE/",p2,"/IMG_DATA/",i2[4]))
b8<-rast(paste0(outDir,"/sentinel-2/",sent2$record_id,".SAFE/GRANULE/",p2,"/IMG_DATA/",i2[5]))

#b_ml<-c(b2,b3,b4,b8)
b_ml<-c(b2,b3,b4)

#add Areo of interest (to be created in GIS)
ad<-read_sf(paste0(outDir,"/shp/Bersia.shp"))
ad<-st_transform(ad,crs = crs(b_ml))
b_ml_mask<-crop(b_ml,ad)
b_ml_mask<-mask(b_ml_mask,ad)

b_ml4326<-project(b_ml_mask, "epsg:4326")#change projection system

#write the 4 layer raster
writeRaster(b_ml4326,paste0(outDir,"/ortho/Bersia_s2.tif"),overwrite=T)


###adding an additional layer from sentinel1

#before running these lines you must correct the sentinal 1 data using the python script. 

s2<-rast(paste0(outDir,"/ortho/Temengor_s2.tif"))
s1<-rast(paste0(outDir,"/ortho_s1/Temengor_s1.tif"))

s1<-crop(s1,s2)
s1<-resample(s1,s2)
s_m<-c(s2,s1)


writeRaster(s_m,paste0(outDir,"/ortho/Temengor_s2_s1.tif"),overwrite=T)
