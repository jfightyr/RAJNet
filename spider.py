import time
import pymysql
import json
import traceback
import requests


##################爬虫独立模块######################
# 功能说明：
#   ①爬取并插入各个地区历史疫情数据
#   ②爬取并插入全国历史疫情统计数据
#   ③爬取并插入百度新闻标题最新数据
# 文件说明：
#   ①可独立运行呈现结果
#   ②调用online方法运行
####################################################
def get_tencent_data():
    url1 = 'https://api.inews.qq.com/newsqa/v1/query/inner/publish/modules/list?modules=diseaseh5Shelf'
    url2 = "https://api.inews.qq.com/newsqa/v1/query/inner/publish/modules/list?modules=chinaDayList,chinaDayAddList,nowConfirmStatis,provinceCompare"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36"
    }
    r1 = requests.get(url1, headers)
    r2 = requests.get(url2, headers)
    res1 = json.loads(r1.text)  # json字符串转字典
    res2 = json.loads(r2.text)
    data_all1 = res1['data']['diseaseh5Shelf']
    data_all2 = res2['data']

    history = {}    # 历史数据
    for i in data_all2["chinaDayList"]:
        ds = i["y"] + "." + i["date"]
        tup = time.strptime(ds, "%Y.%m.%d")  # 匹配时间，为了和数据库兼容（01.23-->01-23）
        ds = time.strftime("%Y-%m-%d", tup)   # 改变时间格式,不然插入数据库会报错，数据库是datetime类型
        confirm = i["confirm"]
        suspect = i["suspect"]
        heal = i["heal"]
        dead = i["dead"]
        history[ds] = {"confirm": confirm, "suspect": suspect, "heal": heal, "dead": dead}
    for i in data_all2["chinaDayAddList"]:
        ds = i["y"] + "." + i["date"]
        tup = time.strptime(ds, "%Y.%m.%d")  # 匹配时间
        ds = time.strftime("%Y-%m-%d", tup)  # 改变时间格式
        confirm = i["confirm"]
        suspect = i["suspect"]
        heal = i["heal"]
        dead = i["dead"]
        if ds in history:
        	history[ds].update({"confirm_add": confirm, "suspect_add": suspect, "heal_add": heal, "dead_add": dead})

    details = []  # 当日详细数据
    update_time = data_all1["lastUpdateTime"]
    data_country = data_all1["areaTree"]   # list 之前有25个国家,现在只有中国
    data_province = data_country[0]["children"]  # 中国各省
    for pro_infos in data_province:
        province = pro_infos["name"]  #省名
        for city_infos in pro_infos["children"]:
            city = city_infos["name"]  #城市名
            confirm = city_infos["total"]["confirm"] #累计确诊
            confirm_add = city_infos["today"]["confirm"] #新增确诊
            heal = city_infos["total"]["heal"]  #累计治愈
            dead = city_infos["total"]["dead"]  #累计死亡
            details.append([update_time, province, city, confirm, confirm_add, heal, dead])
    return history, details


def get_conn():
    # 建立数据库连接
    conn = pymysql.connect(host="127.0.0.1", user="root", password="root", db="yiqing", charset="utf8")
    # 创建游标
    cursor = conn.cursor()
    return conn, cursor


def close_conn(conn, cursor):
    if cursor:
        cursor.close()
    if conn:
        conn.close()


# 插入地区疫情历史数据
# 插入全国疫情历史数据
def update_history():
    cursor = None
    conn = None
    try:
        dic, li = get_tencent_data()  # 1代表最新数据
        conn, cursor = get_conn()
        sql = "insert into details(update_time,province,city,confirm,confirm_add,heal,dead) values(%s,%s,%s,%s,%s,%s,%s)"
        sql_query = 'select %s=(select update_time from details order by id desc limit 1)'

        # 对比当前最大时间戳
        cursor.execute(sql_query, li[0][0])

        if not cursor.fetchone()[0]:
            print(f"[INFO] {time.asctime()}  地区历史疫情爬虫已启动,正在获取数据....")
            for item in li:
                print(f"[INFO] {time.asctime()} 已获取地区历史疫情数据：", item)
                cursor.execute(sql, item)
            conn.commit()
            print(f"[INFO] {time.asctime()}  地区历史疫情爬虫已完成，更新到最新数据成功...")
        else:
            print(f"[WARING] {time.asctime()}地区历史疫情爬虫已启动,已是最新数据...")
        dic = get_tencent_data()[0]  # 0代表历史数据字典
        print(f"[INFO] {time.asctime()}  全国历史疫情爬虫已启动，正在获取数据....")
        conn, cursor = get_conn()
        sql = "insert into history values (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        sql_query = "select confirm from history where ds=%s"
        for k, v in dic.items():
            if not cursor.execute(sql_query, k):
                print(f"[INFO] {time.asctime()} 已获取全国历史疫情：",
                      [k, v.get("confirm"), v.get("confirm_add"), v.get("suspect"),
                       v.get("suspect_add"), v.get("heal"), v.get("heal_add"),
                       v.get("dead"), v.get("dead_add")])
                cursor.execute(sql, [k, v.get("confirm"), v.get("confirm_add"), v.get("suspect"),
                                     v.get("suspect_add"), v.get("heal"), v.get("heal_add"),
                                     v.get("dead"), v.get("dead_add")])
        conn.commit()
        print(f"[INFO] {time.asctime()}  全国历史疫情爬虫已完成，更新到最新数据成功...")
    except:
        traceback.print_exc()
    finally:
        close_conn(conn, cursor)


# 爬取百度热搜数据
def get_baidu_hot():
    """
    :return: 返回百度疫情热搜
    """
    # url = "https://voice.baidu.com/act/virussearch/virussearch?from=osari_map&tab=0&infomore=1"
    url = "https://top.baidu.com/board?tab=realtime"
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    }
    res = requests.get(url, headers=headers)
    # res.encoding = "gb2312"
    html = res.text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html,features="html.parser")
    kw = soup.select("div.c-single-text-ellipsis")
    count = soup.select("div.hot-index_1Bl1a")
    context = []
    for i in range(len(kw)):
        k = kw[i].text.strip()   #移除左右空格
        v = count[i].text.strip()
        #         print(f"{k}{v}".replace('\n',''))
        context.append(f"{k}{v}".replace('\n', ''))
    return context


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

# 插入百度热搜实时数据
def update_hotsearch():
    cursor = None
    conn = None
    try:
        print(f"[INFO] {time.asctime()}  新闻资讯爬虫已启动，正在获取数据...")
        context = get_baidu_hot()
        conn, cursor = get_conn()
        sql = "insert into hotsearch(dt,content) values(%s,%s)"
        ts = time.strftime("%Y-%m-%d %X")
        for i in context:
            print(f"[INFO] {time.asctime()}  已获取历史疫情:", [ts, i])
            cursor.execute(sql, (ts, i))  # 插入数据
        conn.commit()  # 提交事务保存数据
        print(f"[INFO] {time.asctime()}  新闻资讯爬虫已完成，更新到最新数据成功...")
    except:
        traceback.print_exc()
    finally:
        close_conn(conn, cursor)


def online():
    update_history()
    update_hotsearch()
    return 200


if __name__ == "__main__":
    update_history()
    update_hotsearch()
