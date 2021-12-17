url=hadoop@ip-172-22-136-126.ec2.internal
ssh -i ~/emr_key.pem $url

#sudo python3 -m pip install pandas
#to launch EMR cluster
pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
pyspark --packages $pkg_list --num-executors 10 --conf "spark.executor.memoryOverhead=2048" --executor-memory 9g --conf "spark.driver.memoryOverhead=6144" --driver-memory 50g --executor-cores 3 --driver-cores 5 --conf "spark.default.parallelism=60" --conf "spark.sql.shuffle.partitions=60" --conf "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2"

#to launch Sparkm local cloud9
pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
pyspark --packages $pkg_list  --num-executors 3 --executor-memory 9G --executor-cores 2


from pyspark import SparkContext, SparkConf, HiveContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import concat, col, lit
import csv
import pandas as pd
import numpy as np
from pyspark.sql.window import Window
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

segments = ['starting_family', 'nuclear_family', 'extended_family', 'singles', 'couples', 'empty_nesters', 'housemates']

##################################################################################################################
##########################                    Data preparation                      ##############################
##################################################################################################################
#Reading segmented household data
hh_df = spark.read.parquet('s3a://ada-dev/xindee/202108/telco_households/segments/new/*.parquet').cache()

########################################

#Reading connection data
month = '2021{04,05}'
telco_df = spark.read.parquet('s3a://ada-geogrid-feature-set/telco/MY/'+month+'_new_logic_tmi_mcc/*.parquet')\
        .select('ifa', 'device_year_released', 'platform', 'device_vendor', 'device_category', 'device_model', 'cel_connections_count','wifi_connections_count', explode('status_quintuples'))\
        .select('ifa','device_year_released', 'platform', 'device_vendor','device_category', 'device_model', 'cel_connections_count','wifi_connections_count', 'col.*')

telco_df = spark.read.parquet('s3a://ada-geogrid-feature-set/telco/MY/'+month+'_new_logic_tmi_mcc/*.parquet')\
        .select('ifa', explode('status_quintuples'))\
        .select('ifa','col.*').cache()


###Getting data for device age estimation
month_dict = {'january':'01','february':'02','march':'03','april':'04','may':'05','june':'06','july':'07','august':'08','september':'09','october':'10','november':'11','december':'12',}
telco_df = telco_df.withColumn('device_year', telco_df.device_year_released.substr(1,4)).cache()
telco_df = telco_df.withColumn('device_month', telco_df.device_year_released.substr(6,9))
telco_df = telco_df.na.replace(month_dict, 1, 'device_month')
telco_df = telco_df.withColumn('year_month', concat(col('device_year'), lit('-'), col('device_month')))
telco_df = telco_df.withColumn('timestamp', (to_timestamp(col('year_month'), 'yyyy-MM')))
telco_df = telco_df.withColumn('device_age',  months_between(current_date(), col('timestamp')))

###Getting data for household wifi availability estimation
joint = hh_df.join(telco_df, on='ifa', how='inner').cache()
has_wifi = joint.filter(col('carrier_con_type')=='WIFI').select('household_id')
no_wifi = joint.join(has_wifi, on='household_id', how='left_anti').withColumn('has_wifi', F.lit(0)) #left anti to filter out households with mixed connection types
hh_df = no_wifi.select('household_id', 'has_wifi').join(hh_df, on='household_id', how='outer') #reflecting back this information to hh_df
hh_df = hh_df.na.fill(value=1, subset=['has_wifi'])

########################################

#Getting affluence data
month = '2021{04,05,06}'
aff_df = spark.read.parquet('s3a://ada-prod-data/etl/data/brq/sub/affluence/monthly/MY/'+month+'/*.parquet').withColumn("filename", input_file_name())
aff_df = aff_df.select('ifa', 'final_score', 'final_affluence', 'state_only', 'filename')
aff_df = aff_df.withColumn("month",F.split(F.col("filename"),"/").getItem(10))
## Ranking by latest month and selecting the latest month per IFA
window_spec = Window.partitionBy('ifa').orderBy(F.col('month').desc())
aff_df = aff_df.withColumn('month_rank',F.row_number().over(window_spec))
aff_df = aff_df.filter(col('month_rank') == 1)
##Inferring the decision maker as richest person in household
aff_df = aff_df.join(hh_df, on='ifa', how='inner')
window_spec = Window.partitionBy('household_id').orderBy(F.col("final_score").desc())
aff_df = aff_df.withColumn('affluence_rank', F.row_number().over(window_spec))
aff_df = aff_df.withColumn('decision_maker', F.when(col('affluence_rank')==1, 1).otherwise(0)).cache()
decision_maker = aff_df.filter(col('decision_maker')==1).cache()
member = aff_df.filter(col('decision_maker')==0).cache()

########################################

#Getting taxonomised app data
app_df = spark.read.parquet('s3a://ada-dev/DA_datamart/dynamic/app/MY/'+month+'/*.parquet').cache()

##################################################################################################################
##########################                      Getting stats                       ##############################
##################################################################################################################
#Basic stats
member.groupBy('segment').agg(F.countDistinct('household_id')).show()
decision_maker.groupBy('segment').agg(F.countDistinct('household_id')).show()
member.groupBy('segment').agg(F.countDistinct('ifa')).show()
decision_maker.groupBy('segment').agg(F.countDistinct('ifa')).show()

#Average household size, top state
for seg in segments:
    print(seg)
    segment_df = hh_df.filter(col('segment')==seg)
    print('Average household sizes')
    segment_df.agg(F.avg('household_size')).show()
    print('Affluence share')
    segment_df = segment_df.select('ifa').join(aff_df, on='ifa', how='inner')
    segment_df.groupBy('final_affluence').agg(F.countDistinct('ifa').alias('count')).sort('count', ascending=False).show(20)
    print('State with highest share')
    segment_df.groupBy('home_state').agg(F.countDistinct('household_id').alias('count')).sort('count', ascending=False).show(20)

for seg in segments:
    print(seg)
    segment_df = hh_df.filter(col('segment')==seg)
    segment_df.filter(col('home_state')!='None').select('household_id').distinct().count()
##################################################################################################################
##################################################################################################################
#Distribution of mobile carriers: decision makers
for seg in segments:
    print(seg)
    segment_df = decision_maker.filter(col('segment')==seg)
    segment_df = segment_df.join(telco_df, on='ifa', how='inner')
    segment_df.filter(col('carrier_con_type')=='CELLULAR').groupBy('carrier').agg(F.countDistinct('ifa').alias('count')).sort('count', ascending=False).show(20)

#Distribution of mobile carriers: members
for seg in segments:
    print(seg)
    segment_df = member.filter(col('segment')==seg)
    segment_df = segment_df.join(telco_df, on='ifa', how='inner')
    segment_df.filter(col('carrier_con_type')=='CELLULAR').groupBy('carrier').agg(F.countDistinct('ifa').alias('count')).sort('count', ascending=False).show(20)

#Distribution of dual simmers
for seg in segments:
    print(seg)
    segment_df = hh_df.filter(col('segment')==seg)
    dual_simmers = telco_df.filter(col('carrier_con_type')=='CELLULAR').select('ifa', 'carrier').distinct()
    dual_simmers = dual_simmers.groupBy('ifa').agg(countDistinct('carrier').alias('sims'))
    dual_simmers = dual_simmers.filter(col('sims') > 1).withColumn('dual_sim', F.lit(1))
    segment_df = segment_df.join(dual_simmers, on='ifa', how='inner')
    segment_df = segment_df.groupBy('household_id').agg(F.sum('dual_sim').alias('sum_dual_sim'))
    segment_df.filter(col('sum_dual_sim')>1)
    segment_df.distinct().count()


##################################################################################################################
##################################################################################################################
#Average device details: decision makers
for seg in segments:
    print(seg)
    segment_df = decision_maker.filter(col('segment')==seg)
    segment_df = segment_df.join(telco_df, on='ifa', how='inner')
    segment_df.agg(F.avg('device_age')).show()
    segment_df.groupBy('device_category').agg(F.countDistinct('ifa').alias('count')).sort(col('device_category'), ascending=True).show()
    segment_df.groupBy('device_vendor').agg(F.countDistinct('ifa').alias('count')).sort(col('count'), ascending=False).show()
    segment_df.groupBy('platform').agg(F.countDistinct('ifa').alias('count')).sort(col('platform'), ascending=True).show()

#Average device details: members
for seg in segments:
    print(seg)
    segment_df = member.filter(col('segment')==seg)
    segment_df = segment_df.join(telco_df, on='ifa', how='inner')
    segment_df.agg(F.avg('device_age')).show()
    segment_df.groupBy('device_category').agg(count('ifa').alias('count')).sort(col('device_category'), ascending=True).show()
    segment_df.groupBy('platform').agg(count('ifa').alias('count')).sort(col('platform'), ascending=True).show()
    segment_df.groupBy('device_vendor').agg(count('ifa').alias('count')).sort(col('count'), ascending=False).show()

##################################################################################################################
##################################################################################################################
#Most popular apps: decision makers
for seg in segments:
    print(seg)
    segment_df = decision_maker.filter(col('segment')==seg)
    segment_df = segment_df.join(app_df, on='ifa', how='inner')
    segment_df.groupBy('app_l1_name').agg(F.sum('brq_count').alias('sum_brq')).sort(col('sum_brq'), ascending=False).show()
    segment_df.agg(F.sum('brq_count')).show()

#Most popular apps: members
for seg in segments:
    print(seg)
    segment_df = member.filter(col('segment')==seg)
    segment_df = segment_df.join(app_df, on='ifa', how='inner')
    segment_df.groupBy('app_l1_name').agg(F.sum('brq_count').alias('sum_brq')).sort(col('sum_brq'), ascending=False).show()
    segment_df.agg(F.sum('brq_count')).show()

##################################################################################################################
##################################################################################################################
#Connection types data
for seg in segments:
    print(seg)
    segment_df = hh_df.filter(col('segment')==seg)
    segment_df = segment_df.join(telco_df, on='ifa', how='inner')
    print('Usage frequency by connection type')
    segment_df.agg(F.sum('wifi_connections_count').alias('wifi_connections_count') , F.sum('cel_connections_count').alias('cel_connections_count')).show()
    segment_df.groupBy('carrier_con_type').agg(F.sum('connection_brq').cast(DecimalType(20))).show() #remove scientific notation
    print('Household ISPs')
    home_users = segment_df.filter( col('carrier_con_type')=='WIFI' ).select('ifa', 'carrier').distinct()
    home_users.groupBy('carrier').agg(F.countDistinct('ifa').alias('count')).sort(col('count'), ascending=False).show()

#Wifi availability data
for seg in segments:
    print(seg)
    segment_df = hh_df.filter((col('segment')==seg) & (col('has_wifi')==0))
    segment_df = segment_df.join(telco_df, on='ifa', how='inner')
    segment_df.select('carrier_con_type').distinct().show()
    print('Distinct IFAs without Wifi')
    segment_df.select('ifa').distinct().count()
    print('Distinct households without Wifi')
    segment_df.select('household_id').distinct().count()
