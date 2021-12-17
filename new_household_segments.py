###################### SSH ######################

#Change Master Public DNS Name

url=hadoop@ip-172-22-143-234.ec2.internal
ssh -i ~/emr_key.pem $url

pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
pyspark --packages $pkg_list --num-executors 25 --conf "spark.executor.memoryOverhead=2048" --executor-memory 9g --conf "spark.driver.memoryOverhead=6144" --driver-memory 50g --executor-cores 3 --driver-cores 5 --conf "spark.default.parallelism=150" --conf "spark.sql.shuffle.partitions=150" --conf "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=2"

####
#pkg_list=com.databricks:spark-avro_2.11:4.0.0,org.apache.hadoop:hadoop-aws:2.7.1
#pyspark --packages $pkg_list


from pyspark import SparkContext, SparkConf, HiveContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pyspark.sql.types as T
import csv
import pandas as pd
import numpy as np
import sys
from pyspark.sql import Window
from pyspark.sql.functions import rank, col
#import geohash2 as geohash
#import pygeohash as pgh
from functools import reduce
from pyspark.sql import *
from pyspark import StorageLevel

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


##################################################################################################################
##################################################################################################################
#Prep
COUNTRY = 'MY'
MONTH = '2021-09*'

hh_df = spark.read.parquet('s3a://ada-platform-components/Household_v1.1/'+COUNTRY+'/household_id/smc_custom/*/*.parquet').select('household_id','household_size','home_state','household_members',explode('household_members').alias('ifa'))
hh_df = hh_df.filter( (col('household_size') >= 1) & (col('household_size') < 10) ).cache()
hh_df.select('household_id').distinct().count() #10,912,528

## FOR ID: path = 's3a://ada-platform-components/Household/'+COUNTRY+'/202104/*.parquet'

age_df = spark.read.parquet('s3a://ada-platform-components/demographics/output/'+COUNTRY+'/age/'+MONTH+'/*.parquet').select('ifa', 'prediction').cache()
age_df.show(5,0)

##################################################################################################################
##################################################################################################################
#Extended family
# Get IFAs Extended Families >4 and <10
ef_hh = hh_df.filter( col('household_size')>5 )
ef_age = age_df.filter( (col('label')=='35-49') | (col('label')=='50+'))

ef_oldest = ef_hh.join(ef_age, on='ifa', how='inner')

ef_all = ef_oldest.select('household_id').join(hh_df, on='household_id', how='inner').distinct()

extended_family = ef_all.withColumn('segment', F.lit('extended_family'))
extended_family = extended_family.select(sorted(extended_family.columns))

##################################################################################################################
##################################################################################################################
#Nuclear family
# Get IFAs Families 35-49 YO, >2 and <10
nf_hh = hh_df.filter( (col('household_size')>2) & (col('household_size')<10) )
nf_age = age_df.filter(col('label')=='35-49')
nf_oldest = nf_hh.join(nf_age, on='ifa', how='inner')

nf_all = nf_oldest.select('household_id').join(hh_df, on='household_id', how='inner').distinct()
nf_all = nf_all.join(ef_all, on='household_id', how='left_anti') #ensuring zero overlap between extended and nuclear family

nuclear_family = nf_all.withColumn('segment', F.lit('nuclear_family'))
nuclear_family = nuclear_family.select(sorted(nuclear_family.columns))


##################################################################################################################
##################################################################################################################
#Starting family
# Get IFAs Families 25-34 YO, >2 and <10
sf_hh = hh_df.filter( (col('household_size')>2) & (col('household_size')<10) )
sf_age = age_df.filter(col('label')=='25-34')
sf_oldest = sf_hh.join(sf_age, on='ifa', how='inner')

sf_all = sf_oldest.select('household_id').join(hh_df, on='household_id', how='inner').distinct()
sf_all = sf_all.join(ef_all, on='household_id', how='left_anti') #ensuring zero overlap between extended and starting family
sf_all = sf_all.join(nf_all, on='household_id', how='left_anti') #ensuring zero overlap between nuclear and starting family

starting_family = sf_all.withColumn('segment', F.lit('starting_family'))
starting_family = starting_family.select(sorted(starting_family.columns))

##################################################################################################################
##################################################################################################################
#Empty nesters
# Get IFAs Empty Nesters, size = 1 or 2, age 35-49 & 50+
en_hh = hh_df.filter( (col('household_size')==1) | (col('household_size')==2) )
en_age = age_df.filter(col('label')=='50+')

en_all = en_hh.join(en_age, on='ifa', how='inner').drop('label').distinct()
empty_nesters = en_all.withColumn('segment', F.lit('empty_nesters'))
empty_nesters = empty_nesters.select(sorted(empty_nesters.columns))

##################################################################################################################
##################################################################################################################
#Housemates
# Get IFAs housemates, size >= 3 to < 6, 18-24 YO
hm_hh = hh_df.filter( (col('household_size')>=3) & (col('household_size')<6) )
hm_age = age_df.filter(col('label')=='18-24')

hh_all = hm_hh.join(hm_age, on='ifa', how='inner').drop('label').distinct()

housemates = hh_all.withColumn('segment', F.lit('housemates'))
housemates = housemates.select(sorted(housemates.columns))

##################################################################################################################
##################################################################################################################
#Couples
# Get IFAs couple, 18-34 YO
cp_hh = hh_df.filter( col('household_size')==2 )
cp_age = age_df.filter( (col('label')=='18-24') | (col('label')=='25-34') )

cp_all = cp_hh.join(cp_age, on='ifa', how='inner').drop('label').distinct()
couples = cp_all.withColumn('segment', F.lit('couples'))
couples = couples.select(sorted(couples.columns))

##################################################################################################################
##################################################################################################################
#Singles
# Get IFAs 18-24 YO, single
sn_hh = hh_df.filter( col('household_size')==1 )
sn_age = age_df.filter(col('label')=='18-24')

sn_all = sn_hh.join(sn_age, on='ifa', how='inner').drop('label').distinct()
singles = sn_all.withColumn('segment', F.lit('singles'))
singles = singles.select(sorted(singles.columns))

##################################################################################################################
##################################################################################################################
#Output
all = starting_family.union(nuclear_family).union(extended_family).union(empty_nesters).union(housemates).union(couples).union(singles)
all = all.distinct().cache()
all.count() #20,297,819
all.select('household_id').distinct().count() #6,432,515
all.select('ifa').distinct().count() #16,264,528

output = 's3a://ada-dev/azeef/new_household/'+COUNTRY+'/202110/segments'
all.write.parquet(output, mode='overwrite')


##################################################################################################################
##################################################################################################################
all.groupBy('segment').agg(F.countDistinct('household_id')).show()
'''
+---------------+-------------------+
|        segment|count(household_id)|
+---------------+-------------------+
|starting_family|             662931|
|        singles|             204157|
|        couples|            2632110|
| nuclear_family|            2332172|
|     housemates|             812566|
|extended_family|             597207|
|  empty_nesters|               7541|
+---------------+-------------------+
'''
all.groupBy('segment').agg(F.countDistinct('ifa')).show()
'''
+---------------+----------+
|        segment|count(ifa)|
+---------------+----------+
|starting_family|   2354335|
|        singles|    200262|
|        couples|   3764611|
| nuclear_family|   7876968|
|     housemates|    851756|
|extended_family|   3995019|
|  empty_nesters|      7009|
+---------------+----------+
'''



##########
