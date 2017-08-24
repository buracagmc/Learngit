#coding:utf-8

import pandas as pd
import pymysql, datetime
from pandas.io.sql import read_sql


class AD_Statistics_select:

    def __init__(self, city, start_time, end_time, platform=None, engine_type=None):
        self.city = city
        self.start_time = start_time
        self.end_time = end_time
        self.platform = platform
        self.engine_type = engine_type

        self.date_max = (datetime.date(int(self.end_time[0:4]), int(self.end_time[5:7]), int(self.end_time[8:10])) -
        datetime.date(int(self.start_time[0:4]), int(self.start_time[5:7]), int(self.start_time[8:10]))).days +1

        self.df = self.SqlToDataframe()
        self.keyword_list = self.df['keyword']



    def SqlToDataframe(self):

        ## 连接到数据库

        into_db = ['g1-sem-db-01.dns.guazi.com', 'gz_bi_sem_admin',
                   'jn#Qza541q3(v@T3n8', 'guazi_bi_sem', '3313', 'utf8']

        cnxn = pymysql.connect(host=into_db[0], user=into_db[1],
                               passwd=into_db[2], db=into_db[3],
                               port=int(into_db[4]), charset=into_db[5])

        if self.platform == None and self.engine_type == None:
            sql = "SELECT  keyword, COUNT(DISTINCT log_date) AS date_count, SUM(present) AS present_total, \
                   SUM(click) AS click_total, SUM(click)/SUM(present) AS CTR,  SUM(uv) AS uv_total, \
                   AVG(IF(price>0, price, NULL)) AS price_avg, AVG(replace(position,'\r','')) AS position_avg, \
                   COUNT(DISTINCT IF(click>0,log_date,NULL)) AS clicked_num \
                   FROM ib_sem_report  \
                   WHERE log_date BETWEEN '{}' AND '{}' \
                   AND city = '{}' \
                   GROUP BY 1 \
                   ORDER BY 1".format(self.start_time, self.end_time, self.city)

        if self.platform != None and self.engine_type == None:
            sql = "SELECT  keyword, COUNT(DISTINCT log_date) AS date_count, SUM(present) AS present_total, \
                   SUM(click) AS click_total, SUM(click)/SUM(present) AS CTR,  SUM(uv) AS uv_total, \
                   AVG(IF(price>0, price, NULL)) AS price_avg, AVG(replace(position,'\r','')) AS position_avg, \
                   COUNT(DISTINCT IF(click>0,log_date,NULL)) AS clicked_num\
                   FROM ib_sem_report  \
                   WHERE log_date BETWEEN '{}' AND '{}' \
                   AND city = '{}' \
                   AND platform = '{}' \
                   GROUP BY 1 \
                   ORDER BY 1".format(self.start_time, self.end_time, self.city, self.platform)

        if self.platform == None and self.engine_type != None:
            sql = "SELECT  keyword, COUNT(DISTINCT log_date) AS date_count, SUM(present) AS present_total, \
                   SUM(click) AS click_total, SUM(click)/SUM(present) AS CTR,  SUM(uv) AS uv_total, \
                   AVG(IF(price>0, price, NULL)) AS price_avg, AVG(replace(position,'\r','')) AS position_avg, \
                   COUNT(DISTINCT IF(click>0,log_date,NULL)) AS clicked_num\
                   FROM ib_sem_report  \
                   WHERE log_date BETWEEN '{}' AND '{}' \
                   AND city = '{}' \
                   AND engine_type = '{}' \
                   GROUP BY 1 \
                   ORDER BY 1".format(self.start_time, self.end_time, self.city, self.engine_type)

        if self.platform != None and self.engine_type != None:
            sql = "SELECT  keyword, COUNT(DISTINCT log_date) AS date_count, SUM(present) AS present_total, \
                   SUM(click) AS click_total, SUM(click)/SUM(present) AS CTR,  SUM(uv) AS uv_total, \
                   AVG(IF(price>0, price, NULL)) AS price_avg, AVG(replace(position,'\r','')) AS position_avg, \
                   COUNT(DISTINCT IF(click>0,log_date,NULL)) AS clicked_num\
                   FROM ib_sem_report  \
                   WHERE log_date BETWEEN '{}' AND '{}' \
                   AND city = '{}' \
                   AND platform = '{}' \
                   AND engine_type = '{}'  \
                   GROUP BY 1 \
                   ORDER BY 1".format(self.start_time, self.end_time, self.city, self.platform, self.engine_type)

        ## 使用pd接口一步到位啊！！ ##
        try:
            frame = read_sql(sql, cnxn)

        except Exception:
            frame = pd.DataFrame([])
        return frame


    def keyword_detail(self, keyword):

        ## 连接到数据库

        into_db = ['g1-sem-db-01.dns.guazi.com', 'gz_bi_sem_admin',
                   'jn#Qza541q3(v@T3n8', 'guazi_bi_sem', '3313', 'utf8']

        cnxn = pymysql.connect(host=into_db[0], user=into_db[1],
                               passwd=into_db[2], db=into_db[3],
                               port=int(into_db[4]), charset=into_db[5])

        if self.platform == None and self.engine_type == None:
            sql = "SELECT  keyword, CAST(REPLACE(position, '\r', '') AS DECIMAL(4,2)) AS position, click/present AS CTR , price \
                           FROM ib_sem_report  \
                           WHERE log_date BETWEEN '{}' AND '{}' \
                           AND city = '{}'  \
                           AND click <> 0   \
                           AND present <> 0  \
                           AND keyword = '{}'  \
                           AND click < present \
                           ORDER BY 1".format(self.start_time, self.end_time, self.city, keyword)

        if self.platform != None and self.engine_type == None:
            sql = "SELECT  keyword, CAST(REPLACE(position, '\r', '') AS DECIMAL(4,2)) AS position, click/present AS CTR , price \
                                       FROM ib_sem_report  \
                                       WHERE log_date BETWEEN '{}' AND '{}' \
                                       AND city = '{}'  \
                                       AND click <> 0   \
                                       AND present <> 0  \
                                       AND platform = '{}' \
                                       AND keyword = '{}'  \
                                       AND click < present \
                                       ORDER BY 1".format(self.start_time, self.end_time, self.city, self.platform, keyword)

        if self.platform == None and self.engine_type != None:
            sql = "SELECT  keyword, CAST(REPLACE(position, '\r', '') AS DECIMAL(4,2)) AS position, click/present AS CTR , price \
                                       FROM ib_sem_report  \
                                       WHERE log_date BETWEEN '{}' AND '{}' \
                                       AND city = '{}'  \
                                       AND click <> 0   \
                                       AND present <> 0  \
                                       AND engine_type = '{}' \
                                       AND keyword = '{}'  \
                                       AND click < present \
                                       ORDER BY 1".format(self.start_time, self.end_time, self.city, self.engine_type, keyword)

        if self.platform != None and self.engine_type != None:
            sql = "SELECT  keyword, CAST(REPLACE(position, '\r', '') AS DECIMAL(4,2)) AS position, click/present AS CTR , price \
                                       FROM ib_sem_report  \
                                       WHERE log_date BETWEEN '{}' AND '{}' \
                                       AND city = '{}'  \
                                       AND click <> 0   \
                                       AND present <> 0  \
                                       AND platform = '{}' \
                                       AND engine_type =  '{}'  \
                                       AND keyword = '{}'  \
                                       AND click < present \
                                       ORDER BY 1".format(self.start_time, self.end_time, self.city, self.platform,
                                                          self.engine_type, keyword)

        ## 使用pd接口一步到位啊！！ ##
        try:
            keyword_detail_frame = read_sql(sql, cnxn)

        except Exception:
            keyword_detail_frame = pd.DataFrame([])

        return keyword_detail_frame







