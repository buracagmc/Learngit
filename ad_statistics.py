#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import csv, os

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.stats import pearsonr
from matplotlib.font_manager import FontProperties
from ggplot import *
from ad_statistics_select import *       # ad_statistics_select 类里是sql查询函数，返回dataframe


font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
if not os.path.exists('keywords_pearson_point'):
    os.mkdir('keywords_pearson_point')

class AD_Statistics(AD_Statistics_select):


    def keyword_SortBy_CTR(self, keyword, detail=False, **kwargs):
        """
        输入需要查询的关键词，默认返回按照转化率排序得出的排序结果
        以及转化率、平均单价、平均投放位置信息；

        如果需要输出所有字段信息，则设置`detail=True`

        param
        ========
        keyword:必须；输入的需要查询的关键词
        detail：逻辑变量，默认为False；是否需要输出该关键词的所有字段信息
        df:是否输入dataframe，默认为所有未筛选过的数据

        return
        ========
        该关键词按照转化率的排序结果
        """
        if 'df' in kwargs.keys():
            df = kwargs['df'].sort_values('CTR', ascending=False)
        else:
            df = self.df.sort_values('CTR', ascengding=False)

        df.index = range(1, len(df) + 1)

        if detail:
            return df.loc[df['keyword'] == keyword, df.columns]

        return df.loc[df['keyword'] == keyword, ['CTR', 'price_avg', 'position_avg']]


    def keyword_SortBy_present(self, keyword, detail=False, **kwargs):
        """
        输入需要查询的关键词，默认返回按照展现次数排序得出的排序结果
        以及转化率、平均单价、平均投放位置信息；

        如果需要输出所有字段信息，则设置`detail=True`

        param
        ========
        keyword:必须；输入的需要查询的关键词
        detail：逻辑变量，默认为False；是否需要输出该关键词的所有字段信息
        df:是否输入dataframe，默认为所有未筛选过的数据

        return
        ========
        该关键词按照展现次数的排序结果
        """
        if 'df' in kwargs.keys():
            df = kwargs['df'].sort_values('present_total', ascending=False)

        else:
            df = self.df.sort_values('present_total', ascending=False)

        df.index = range(1, len(df) + 1)

        if detail:
            return df.loc[df['keyword'] == keyword, df.columns]

        return df.loc[df['keyword'] == keyword, ['CTR', 'price_avg', 'position_avg']]


    def keyword_SortBy_click(self, keyword, detail=False, **kwargs):
        """
        输入需要查询的关键词，默认返回按照点击次数排序得出的排序结果
        以及转化率、平均单价、平均投放位置信息；

        如果需要输出所有字段信息，则设置`detail=True`

        param
        ========
        keyword:必须；输入的需要查询的关键词
        detail：逻辑变量，默认为False；是否需要输出该关键词的所有字段信息
        df:是否输入dataframe，默认为所有未筛选过的数据

        return
        ========
        该关键词按照点击次数的排序结果
        """
        if 'df' in kwargs.keys():
            df = kwargs['df'].sort_values('click_total', ascending=False)
        else:
            df = self.df.sort_values('click_total', ascengding=False)

        df.index = range(1, len(df) + 1)

        if detail:
            return df.loc[df['keyword'] == keyword, df.columns]

        return df.loc[df['keyword'] == keyword, ['CTR', 'price_avg', 'position_avg']]


    def present_top(self, n=100):
        """
        返回展现量前n位的keyword数据，n默认为100
        """
        return self.df.sort_values('present_total', ascending=False).head(n)

    def present_part(self, before=0.81):

        # 返回一个tuple
        # 展现量before前的dataframe，before到1.0的dataframe，1.0(也即无点击的)的dataframe

        click_df = self.df.sort_values('present_total', ascending=False)

        add_rate = []
        s = click_df['present_total'].sum()

        for i in range(1, len(click_df)+1):
            add_rate.append(click_df['present_total'][0:i].sum() / s)

        click_df['p_add_rate'] = add_rate

        #click_df['rate'] = click_df['click_total'] / click_df['click_total'].sum()
        #click_df['rank'] = range(1, len(click_df)+1)
        #click_df.head(200).plot('rank', 'add_rate', kind='line')


        return click_df[click_df['p_add_rate'] < before], click_df[(click_df['p_add_rate']>0.81)&(click_df['present_total']>self.date_max)], \
                click_df[click_df['present_total']<=self.date_max]

    def click_top(self, n=100):
        """
        返回点击量前n位的keyword数据，n默认为100
        """
        return self.df.sort_values('click_total', ascending=False).head(n)

    def click_part(self, before=0.81):

        # 返回一个tuple
        # 点击量before前的dataframe，before到1.0的dataframe，1.0(也即无点击的)的dataframe

        click_df = self.df.sort_values('click_total', ascending=False)

        add_rate = []
        s = click_df['click_total'].sum()

        for i in range(1, len(click_df)+1):
            add_rate.append(click_df['click_total'][0:i].sum() / s)

        click_df['c_add_rate'] = add_rate

        #click_df['rate'] = click_df['click_total'] / click_df['click_total'].sum()
        #click_df['rank'] = range(1, len(click_df)+1)
        #click_df.head(200).plot('rank', 'add_rate', kind='line')


        return click_df[click_df['c_add_rate'] < before],click_df[(click_df['c_add_rate']>0.81)&(click_df['c_add_rate']<1.0)], \
                click_df[click_df['c_add_rate'] == 1.0]

    def price_top(self, n=100):
        """
        返回price_avg前n位的keyword数据，n默认为100
        """
        return self.df.sort_values('price_avg', ascending=False).head(n)

    def date_count_top(self, n=100):
        """
        返回点击量前n位的keyword数据，n默认为100
        """
        return self.df.sort_values('click_total', ascending=False).head(n)

    def CTR_top(self, n=100):
        """
        返回点击量前n位的keyword数据，n默认为100
        """
        return self.df.sort_values('CTR', ascending=False).head(n)

    ################筛选是否有点击的关键词#####################

    def click_none(self):
        # 返回关键词里没有点击量的
        return self.df[self.df['click_total'] == 0]

    def click_not_none(self):
        # 返回关键词里有点击量的
        return self.df[self.df['click_total'] != 0]
    ##########################################################


    ####################show date_count--present_avg#########
    def show_click_none(self):
        show_df = self.click_none()
        show_df['present_avg'] = show_df['present_total'] / show_df['date_count']
        p = ggplot(aes(x='date_count', y='present_avg'), data=show_df) + geom_point()

        p.show()

    def show_click_not_none(self):
        show_df = self.click_top()
        # show_df = self.click_not_none()
        show_df['present_avg'] = show_df['present_total'] / show_df['date_count']
        show_df['click_avg'] = show_df['click_total'] / show_df['date_count']

        fig, axe = plt.subplots(2, 2)

        show_df.plot(x='date_count', y='click_avg', kind='scatter', ax=axe[0, 0], figsize=(5, 5), alpha=0.5)

        show_df.plot(x='date_count', y='present_avg', kind='scatter', ax=axe[0, 1], figsize=(5, 5), alpha=0.5)

        show_df.plot(x='present_avg', y='click_avg', kind='scatter', ax=axe[1, 0], figsize=(5, 5), alpha=0.5)

        # p = ggplot(aes(x='date_count', y='present_avg'), data=show_df) + geom_point()

        # p.show()
        plt.show()

    #########################################################

    def pearson_points(self, keyword_df):

        # 返回keyword的position-price的皮尔逊相关系数以及对应的P值，position-CTR的皮尔逊相关系数以及对应的P值

        grouped_by_position = keyword_df.groupby('position')  # groupby
        avg_grouped_by_position = grouped_by_position['CTR', 'price'].agg([np.mean])  # 求均值
        avg_grouped_by_position['position'] = avg_grouped_by_position.index  # 加index

        array_position = np.array(avg_grouped_by_position['position'])
        array_price = np.array(avg_grouped_by_position['price']).reshape(len(array_position), )
        array_CTR = np.array(avg_grouped_by_position['CTR']).reshape(len(array_position), )

        p_p = pearsonr(array_position, array_price)
        po_C = pearsonr(array_position, array_CTR)
        pr_C = pearsonr(array_price, array_CTR)

        return [keyword_df['keyword'][0], p_p[0], p_p[1], po_C[0], po_C[1], pr_C[0], pr_C[1]]
        '''
        return u"""
        【{}】的皮尔逊相关系数-----P值
        position-price:{},{}
        position-CTR:{}, {}
        price-CTR:{},{}""".format(keyword_df['keyword'][0], p_p[0], p_p[1], po_C[0], po_C[1], pr_C[0], pr_C[1])
        '''

    def keyword_show(self, keyword_df):

        """
        根据keyword绘出其price-position、CTR-position的关系图
        """
        grouped_by_position = keyword_df.groupby('position')       # groupby
        avg_grouped_by_position = grouped_by_position['CTR', 'price'].agg([np.mean])  # 求均值
        avg_grouped_by_position['position'] = avg_grouped_by_position.index

        array_position = np.array(avg_grouped_by_position['position'])
        array_price = np.array(avg_grouped_by_position['price']).reshape(len(array_position), )
        array_CTR = np.array(avg_grouped_by_position['CTR']).reshape(len(array_position), )

        fig = plt.figure()
        fig.suptitle(u'%s' % keyword_df['keyword'][0], fontproperties=font)

        ax1 = fig.add_subplot(2, 2, 1)
        p_p = pearsonr(array_position, array_price)
        ax1.set_title(u'pearson系数：%s,伴随P值:%s' % (p_p[0], p_p[1]), fontproperties=font)


        ax2 = fig.add_subplot(2, 2, 3)
        po_C = pearsonr(array_position, array_CTR)
        ax2.set_title(u'pearson系数：%s,伴随P值:%s' % (po_C[0], po_C[1]), fontproperties=font)


        ax3 = fig.add_subplot(1, 2, 2)
        pr_C = pearsonr(array_price, array_CTR)
        ax3.set_title(u'pearson系数：%s,伴随P值:%s' % (pr_C[0], pr_C[1]), fontproperties=font)



        avg_grouped_by_position.plot(x='position', y='price', kind='scatter', ax=ax1, figsize=(12, 6),
                                     alpha=0.5, color='#33CC99', sharex=True)
        avg_grouped_by_position.plot(x='position', y='CTR',   kind='scatter', ax=ax2, figsize=(12, 6),
                                     alpha=0.5, color='#FF3366')
        avg_grouped_by_position.plot(x='price', y='CTR', kind='scatter', ax=ax3, alpha=0.5, color='#FF4500')

        return plt

    def keyword_list_show(self, keyword_list):

        # yield返回在list中每一个keyword的可视化图

        for i in keyword_list:
            keyword_df = self.keyword_detail(i)
            yield self.keyword_show(keyword_df)

    def keyword_list_pearson(self, keyword_list):

        # 返回list中每一个keyword的详细pearson系数及伴随P值信息


        result = []
        for i in keyword_list:
            keyword_df = self.keyword_detail(str(i.encode('utf-8')))
            result.append(self.pearson_points((keyword_df)))


        pp_df = pd.DataFrame(result)
        pp_df.columns = [u'keyword', u'position-price_pearson', u'position-price_pvalue',
                         u'position-CTR_pearson', u'position-CTR_pvalue', u'price-CTR_pearson', u'price-CTR_pvalue']

        #pp_df.to_csv('keywords_pearson_point/%s-pearson.csv') % self.city
        return pp_df

    def output(self, keyword_list, output_name='default_name'):

        # 输出list中每个keyword的信息到本地

        output = []
        for i in keyword_list:
            keyword_df = self.keyword_detail(str(i.encode('utf-8')))
            output.append(self.pearson_points(keyword_df))

        ## 写入CSV
        with open('keywords_pearson_point/%s.csv' % output_name, 'w', encoding='utf8', newline='') as outfile:
            spamwriter = csv.writer(outfile)
            file_title = ['keyword', 'positon-price皮尔逊系数', '伴随P值', 'position-CTR皮尔逊系数', '伴随P值',
                          'price-CTR皮尔逊系数', '伴随P值']
            spamwriter.writerow(file_title)
            for item in output:
                    spamwriter.writerow(item)
        outfile.close()
        print(u'输出成功')

    # TODO    评分系统待做
    def keyword_score(self, keyword_dataframe):

        """
        =================================================================================================
        构建评分系统，对每个keyword进行得分排名

        # 日均展现量    ==> 表明搜索意愿的大小，越大得分越高
        # CTR          ==> 表明展现到转化的比例，越大越好
        # price_avg    ==> 表明平均价格，越小越好
        # 有点击的天数占总时间区间的比例  ==> 表明了活跃程度，越大越好
        =================================================================================================

        keyword_dataframe: 跟据present_part()或者click_part()获得的去除头部密集部分和尾部稀疏部分后的dataframe
        详细筛选规则请看present_part()函数和click_part()函数

        return
        -------
        keyword_dataframe里每个keyword的排名情况
        """

        #keyword_dataframe['present_mean_by_day'] = keyword_dataframe['present_total'] / keyword_dataframe['date_count']
        #keyword_dataframe['present_mean_rate'] = keyword_dataframe['present_mean_by_day'] / keyword_dataframe['present_mean_by_day'].sum()
        #keyword_dataframe['clicked_rate'] = keyword_dataframe['clicked_num'] / keyword_dataframe['date_count']
       # keyword_dataframe['score'] = keyword_dataframe['present_total'] / self.date_max + keyword_dataframe['CTR'] +\
                                     #(keyword_dataframe['click_total'] / self.date_max) / keyword_dataframe['price_avg']

        keyword_dataframe['click_mean'] = keyword_dataframe['click_total'] / self.date_max
        keyword_dataframe['score'] = keyword_dataframe['click_mean'] / keyword_dataframe['price_avg']

        return keyword_dataframe.sort_values('score', ascending=False)


    def keyword_person_show(self, keyword_list):

        pp_df = self.keyword_list_pearson(keyword_list)
        pp_df.plot(u'position-price_pearson', u'position-CTR_pearson', kind='scatter', alpha=0.5)
        plt.show()

    ########对keyword列表里的keyword做聚类分析并输出可视化结果#########
    def keyword_pearson_kmeans(self, keyword_list):

        ## 将聚类结果可视化呈现出来

        pp_df = self.keyword_list_pearson(keyword_list)
        df_train = np.array(pp_df[['position-price_pearson', 'position-CTR_pearson']])

        kmeans_model = KMeans(n_clusters=4).fit(df_train)

        #  print kmeans_model.labels_:每个点对应的标签值

        colors = ['#33CC99', '#FF3366', '#FF4500', '#00BFFF', 'm', 'y', 'k', 'b']   # 颜色
        markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']  # 数据标识

        plt.figure(figsize=(16, 8))
        ## 将每个点画出
        for i, l in enumerate(kmeans_model.labels_):
            plt.plot(df_train[i][0], df_train[i][1], color=colors[l],
                     marker=markers[l], ls='None', alpha=0.5)

            plt.text(df_train[i][0], df_train[i][1], '%s'%pp_df['keyword'][i], fontproperties=font, fontsize=6)

        #for a,b in zip(df_train['position-price_pearson'], df_train['position-CTR_pearson']):
         #   plt.text(a,b,'%s'%)

        plt.title(u'K = 4, 轮廓系数 = %.03f' %
                      metrics.silhouette_score(df_train, kmeans_model.labels_, metric='euclidean')
                      , fontproperties=font)
        plt.xlabel(u'position-price皮尔逊系数', fontproperties=font)
        plt.ylabel(u'position-CTR皮尔逊系数', fontproperties=font)
        plt.savefig('ad_kmeans1.png', dpi=500)
        plt.show()

    ########进行LR回归分类##########
    # TODO
    def logRegression(self, keyword_dataframe):
        pass


    # TODO 根据
    def regres(self, keyword_dataframe):

        ## 对click排名后的中部
        ## 构建属性has_city_name：是否包含城市名称
        ## 构建属性has_space:    是否包含空格
        worddf = self.click_part()[1]

        f1 = lambda x: 1 if self.city.decode('utf-8') in x else 0
        w_has_cityname = map(f1, worddf['keyword'])

        f2 = lambda x: 1 if ' ' in x else 0
        w_has_space = map(f2, worddf['keyword'])

        worddf = pd.DataFrame(worddf)

        worddf['has_space'] = np.array(w_has_space)
        worddf['city_name'] = np.array(w_has_cityname)
        return worddf






if __name__ == "__main__":

    '''
    ## 交互输入 ##
    city = raw_input(u'请输入需要查询的城市：')
    start_time = raw_input(u'请输入开始时间：')
    end_time = raw_input(u'请输入结束时间：')



    f_p = lambda a: None if a == '' else 3
    platform = f_p(raw_input(u"""请输入平台id:
            其中，1为PC,2位WAP；直接enter则默认为所有平台"""))

    print platform


    f_e = lambda a: None if a == '' else 3
    engine_type = f_e(raw_input(u"""请输入搜索引擎id：
            其中，1为baidu，2为sougou，3为360,4为shenma；直接enter则默认为所有搜索引擎"""))
    print engine_type


    '''

    AD_STA = AD_Statistics('天津', '2017-05-01', '2017-07-20')
    df = AD_STA.SqlToDataframe()
    print len(df)

    ret = AD_STA.click_part()

    print len(ret[0])
    print len(ret[1])
    print len(ret[2])
    #print ret[1]

    '''
    ret[1].plot(u'click_total', u'uv_total', kind='scatter', alpha=0.5)
    plt.show()

    ret[1].plot(u'click_total', u'clicked_num', kind='scatter', alpha=0.5)
    plt.show()

    ret[1].plot(u'clicked_num', u'date_count', kind='scatter', alpha=0.5)
    plt.show()


    ret[1].plot(u'clicked_num', u'present_total', kind='scatter', alpha=0.5)
    plt.show()

    ret[1].plot(u'date_count', u'present_total', kind='scatter', alpha=0.5)
    plt.show()
    '''

    #print ret[1]
    '''
    ################ 构建新属性列 #################
    worddf = ret[1]['keyword']
    print type(worddf)
    print AD_STA.city
    f1 = lambda x: 1 if AD_STA.city.decode('utf-8') in x else 0
    w_has_cityname = map(f1, worddf)


    f2 = lambda x: 1 if ' ' in x else 0
    w_has_space = map(f2, worddf)

    worddf = pd.DataFrame(worddf)
    print worddf
    print type(worddf)
    worddf['has_space'] = np.array(w_has_space)
    worddf['city_name'] = np.array(w_has_cityname)
    print worddf
    #print w_has_cityname

'''

    '''
    keyword_list = ['二手车评估', '网上车市深圳', '人人汽车网']
    picture = AD_STA.keyword_list_show(keyword_list)

    for i in picture:
        print i.show()



        #listt = AD_STA.keyword_list_show(ret[0]['keyword'])

    '''

    # print AD_STA.keyword_pearson_kmeans(ret[0]['keyword'])

    # keyword_df = AD_STA.keyword_detail('瓜子二手车直卖网')
    # print AD_STA.keyword_show(keyword_df).show()
    #print ret[1]

    #print AD_STA.keyword_score(ret[1])[['keyword', 'score', 'CTR', 'price_avg', 'click_mean']]
    #print ret[2]
    #for i in click_top_keyword_list[15:16]:
        #keyword_df = AD_STA.keyword_detail(str(i.encode('utf-8')))
        #print(keyword_df)

    #rr = AD_STA.keyword_list_pearson(click_top_keyword_list[10:15])

    #print AD_STA.keyword_pearson_kmeans(ret[0]['keyword'])
    #print AD_STA.keyword_list_pearson(ret[0]['keyword'])
    #print rr
    #keyword_list = ['重庆市二手车交易网']
    #keyword_df = AD_STA.keyword_detail('瓜子二手车直卖网')

    #print AD_STA.keyword_show(keyword_df).show()


   #AD_STA.output(click_top_keyword_list[15:20])












